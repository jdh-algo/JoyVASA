import json
import os
import io
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings
import torch.nn.functional as F
from logger import logger
torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')
import json
import os
import io
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings
import torch.nn.functional as F
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TalkingHeadDataset')

torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class TalkingHeadDatasetHungry(data.Dataset):
    def __init__(self, root_dir, motion_filename="talking_face.pkl", motion_templete_filename="motion_templete.pkl", 
                 split="train", coef_fps=25, n_motions=100, crop_strategy="random", normalize_type="mix"):
        logger.info(f"Initializing dataset with root_dir: {root_dir}, split: {split}")
        
        self.save_visualizations = False
        
        self.templete_dir = os.path.join(root_dir, motion_templete_filename)
        logger.debug(f"Loading template from: {self.templete_dir}")
        self.templete_dict = pickle.load(open(self.templete_dir, 'rb'))
        
        self.motion_dir = os.path.join(root_dir, motion_filename)
        logger.debug(f"Loading motion data from: {self.motion_dir}")
        
        self.eps = 1e-9
        self.normalize_type = normalize_type
        logger.info(f"Using normalization type: {normalize_type}")

        if split == "train":
            self.root_dir = os.path.join(root_dir, "train.json")
        else:
            self.root_dir = os.path.join(root_dir, "test.json")
        
        logger.debug(f"Loading JSON data from: {self.root_dir}")
        with open(self.root_dir, 'r') as f:
            json_data = json.load(f)
        self.all_data = json_data
        
        logger.debug("Loading motion data...")
        self.motion_data = pickle.load(open(self.motion_dir, "rb"))
        logger.info("Motion data loading complete")

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy
        
        logger.info(f"Dataset initialized with {len(self.all_data)} samples")
        logger.debug(f"Configuration - FPS: {coef_fps}, n_motions: {n_motions}, crop_strategy: {crop_strategy}")
        
    def __len__(self):
        return len(self.all_data)
    
    def check_motion_length(self, motion_data):
        seq_len = motion_data["n_frames"]
        logger.debug(f"Checking motion length: {seq_len} frames")
        
        if seq_len > self.coef_total_len + 2:
            logger.debug("Motion length sufficient")
            return motion_data
        else:
            logger.warning(f"Motion length insufficient ({seq_len}), extending sequence")
            exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
            for frame_index in range(motion_data["n_frames"]):
                exp_list.append(motion_data["motion"][frame_index]["exp"])
                t_list.append(motion_data["motion"][frame_index]["t"])
                scale_list.append(motion_data["motion"][frame_index]["scale"])
                pitch_list.append(motion_data["motion"][frame_index]["pitch"])
                yaw_list.append(motion_data["motion"][frame_index]["yaw"])
                roll_list.append(motion_data["motion"][frame_index]["roll"])

            repeat = 0 
            while len(exp_list) < self.coef_total_len + 2:
                exp_list = exp_list * 2
                t_list = t_list * 2
                scale_list = scale_list * 2
                pitch_list = pitch_list * 2
                yaw_list = yaw_list * 2
                roll_list = roll_list * 2
                repeat += 1
            
            logger.debug(f"Sequence extended {repeat} times")
            
            motion_new = {"motion": []}
            for i in range(len(exp_list)):
                motion = {
                    "exp": exp_list[i],
                    "t": t_list[i],
                    "scale": scale_list[i],
                    "pitch": pitch_list[i],
                    "yaw": yaw_list[i],
                    "roll": roll_list[i],
                }
                motion_new["motion"].append(motion)
            motion_new["n_frames"] = len(exp_list)
            motion_new["repeat"] = repeat
            
            logger.debug(f"New motion length: {motion_new['n_frames']} frames")
            return motion_new
    
    def __getitem__(self, index):
        logger.debug(f"Getting item at index: {index}")
        has_valid_audio = False
        attempts = 0
        max_attempts = 5  # Prevent infinite loops
        
        while not has_valid_audio and attempts < max_attempts:
            attempts += 1
            try:
                # read motion
                metadata = self.all_data[index]
                logger.debug(f"Processing audio file: {metadata['audio_name']}")
                
                motion_data = self.motion_data[metadata["audio_name"]]
                motion_data = self.check_motion_length(motion_data)
                
                # crop audio and coef, count start_frame
                seq_len = motion_data["n_frames"]
                if self.crop_strategy == 'random':
                    end = seq_len - self.coef_total_len
                    if end < 0:
                        logger.warning(f"Invalid sequence length for {os.path.basename(metadata['audio_name'])}, n_frames: {seq_len}")
                        has_valid_audio = False 
                        continue
                    start_frame = np.random.randint(0, seq_len - self.coef_total_len)
                    logger.debug(f"Random crop - start_frame: {start_frame}")
                elif self.crop_strategy == 'begin':
                    start_frame = 0
                elif self.crop_strategy == 'end':
                    start_frame = seq_len - self.coef_total_len
                else:
                    logger.error(f"Unknown crop strategy: {self.crop_strategy}")
                    raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
                
                end_frame = start_frame + self.coef_total_len
                logger.debug(f"Frame range: {start_frame} to {end_frame}")

                # Process motion data
                logger.debug("Processing motion features")
                exp, scale, t, pitch, yaw, roll = [], [], [], [], [], []
                for frame_idx in range(motion_data["n_frames"]):
                    exp.append((motion_data['motion'][frame_idx]["exp"].flatten() - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps))
                    scale.append((motion_data['motion'][frame_idx]["scale"].flatten() - self.templete_dict["mean_scale"]) / (self.templete_dict["std_scale"] + self.eps))
                    t.append((motion_data['motion'][frame_idx]["t"].flatten() - self.templete_dict["mean_t"]) / (self.templete_dict["std_t"] + self.eps))
                    pitch.append((motion_data['motion'][frame_idx]["pitch"].flatten() - self.templete_dict["mean_pitch"]) / (self.templete_dict["std_pitch"] + self.eps))
                    yaw.append((motion_data['motion'][frame_idx]["yaw"].flatten() - self.templete_dict["mean_yaw"]) / (self.templete_dict["std_yaw"] + self.eps))
                    roll.append((motion_data['motion'][frame_idx]["roll"].flatten() - self.templete_dict["mean_roll"]) / (self.templete_dict["std_roll"] + self.eps))

                # Process coefficients
                logger.debug("Processing coefficients")
                coef_keys = ["exp", "pose"]
                coef_dict = {k: [] for k in coef_keys}
                audio = []
                
                for frame_idx in range(start_frame, end_frame):
                    for coef_key in coef_keys:
                        if coef_key == "exp":
                            if self.normalize_type == "mix":
                                normalized_exp = (motion_data['motion'][frame_idx]["exp"].flatten() - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps)
                            else:
                                logger.error("Invalid normalization type for exp")
                                raise RuntimeError("error")
                            coef_dict[coef_key].append([normalized_exp, ])
                        elif coef_key == "pose":
                            if self.normalize_type == "mix":
                                pose_data = np.concatenate((
                                    (motion_data['motion'][frame_idx]["scale"].flatten() - self.templete_dict["min_scale"]) / (self.templete_dict["max_scale"] - self.templete_dict["min_scale"] + self.eps),
                                    (motion_data['motion'][frame_idx]["t"].flatten() - self.templete_dict["min_t"]) / (self.templete_dict["max_t"] - self.templete_dict["min_t"] + self.eps),
                                    (motion_data['motion'][frame_idx]["pitch"].flatten() - self.templete_dict["min_pitch"]) / (self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"] + self.eps),
                                    (motion_data['motion'][frame_idx]["yaw"].flatten() - self.templete_dict["min_yaw"]) / (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"] + self.eps),
                                    (motion_data['motion'][frame_idx]["roll"].flatten() - self.templete_dict["min_roll"]) / (self.templete_dict["max_roll"] - self.templete_dict["min_roll"] + self.eps),
                                ))
                            else:
                                logger.error("Invalid normalization type for pose")
                                raise RuntimeError("pose data error")

                            coef_dict[coef_key].append([pose_data, ])
                        else:
                            logger.error(f"Invalid coefficient key: {coef_key}")
                            raise RuntimeError(f"coef_key error: {coef_key}")
                        
                coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
                
                if coef_dict['exp'].shape[0] != self.coef_total_len:
                    logger.error(f"Invalid coefficient length: {coef_dict['exp'].shape[0]}")
                    raise AssertionError(f'Invalid coef length: {coef_dict["exp"].shape[0]}')

                # Load and process audio
                logger.debug(f"Loading audio from: {metadata['audio_name']}")
                audio_path = metadata["audio_name"]
                audio_clip, sr = torchaudio.load(audio_path)
                audio_clip = audio_clip.squeeze()
                
                if "repeat" in motion_data:
                    logger.debug(f"Repeating audio {motion_data['repeat']} times")
                    for _ in range(motion_data["repeat"]):
                        audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

                if sr != 16000:
                    logger.error(f"Invalid sampling rate: {sr}")
                    raise AssertionError(f'Invalid sampling rate: {sr}')
                
                audio.append(audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)])
                audio = torch.cat(audio, dim=0)
                
                if not (audio.shape[0] == self.coef_total_len * self.audio_unit):
                    logger.warning(f"Invalid audio length: {audio.shape[0]}, expected: {self.coef_total_len * self.audio_unit}")
                    has_valid_audio = False 
                    continue

                # Extract final pairs
                logger.debug("Extracting final pairs")
                keys = ['exp', 'pose']
                audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
                coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},
                            {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]
                
                has_valid_audio = True
                logger.debug("Successfully processed item")
                return audio_pair, coef_pair
                
            except Exception as e:
                logger.error(f"Error processing item {index}: {str(e)}", exc_info=True)
                has_valid_audio = False
                
        if not has_valid_audio:
            logger.error(f"Failed to get valid item after {max_attempts} attempts")
            raise RuntimeError(f"Failed to get valid item at index {index}")

import json
import os
import io
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings
import torch.nn.functional as F
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TalkingHeadDataset')

torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class TalkingHeadDatasetHungry(data.Dataset):
    def __init__(self, root_dir, motion_filename="talking_face.pkl", motion_templete_filename="motion_templete.pkl", 
                 split="train", coef_fps=25, n_motions=100, crop_strategy="random", normalize_type="mix"):
        logger.info(f"Initializing dataset with root_dir: {root_dir}, split: {split}")
        
        self.templete_dir = os.path.join(root_dir, motion_templete_filename)
        logger.debug(f"Loading template from: {self.templete_dir}")
        self.templete_dict = pickle.load(open(self.templete_dir, 'rb'))
        
        self.motion_dir = os.path.join(root_dir, motion_filename)
        logger.debug(f"Loading motion data from: {self.motion_dir}")
        
        self.eps = 1e-9
        self.normalize_type = normalize_type
        logger.info(f"Using normalization type: {normalize_type}")

        if split == "train":
            self.root_dir = os.path.join(root_dir, "train.json")
        else:
            self.root_dir = os.path.join(root_dir, "test.json")
        
        logger.debug(f"Loading JSON data from: {self.root_dir}")
        with open(self.root_dir, 'r') as f:
            json_data = json.load(f)
        self.all_data = json_data
        
        logger.debug("Loading motion data...")
        self.motion_data = pickle.load(open(self.motion_dir, "rb"))
        logger.info("Motion data loading complete")

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy
        
        logger.info(f"Dataset initialized with {len(self.all_data)} samples")
        logger.debug(f"Configuration - FPS: {coef_fps}, n_motions: {n_motions}, crop_strategy: {crop_strategy}")
        

    def visualize_expression(self, expression_data: np.ndarray, title: str = "Expression Visualization") -> None:
        """
        Visualize expression data with a heatmap.
        
        Args:
            expression_data: Normalized expression data (N x D)
            title: Title for the plot
        """
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import os

        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot heatmap
        plt.imshow(expression_data.reshape(-1, expression_data.shape[-1]),
                aspect='auto',
                cmap='RdBu_r',
                vmin=-3,  # Adjust based on your normalization
                vmax=3)
        
        plt.colorbar(label='Normalized Expression Value')
        plt.title(title)
        plt.xlabel('Expression Dimension')
        plt.ylabel('Frame')
        
        # Save plot
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(f'visualizations/{title.lower().replace(" ", "_")}.png', 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()

    def visualize_expression_sequence(self, motion_data, start_idx: int, n_frames: int, 
                                    save_dir: str = 'visualizations') -> None:
        """
        Visualize a sequence of expressions.
        
        Args:
            motion_data: Motion data dictionary
            start_idx: Starting frame index
            n_frames: Number of frames to visualize
            save_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract and normalize expressions
        expressions = []
        for frame_idx in range(start_idx, start_idx + n_frames):
            if frame_idx >= len(motion_data['motion']):
                break
                
            exp = motion_data['motion'][frame_idx]["exp"].flatten()
            normalized_exp = (exp - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps)
            expressions.append(normalized_exp)
        
        expressions = np.stack(expressions)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot heatmap
        im1 = ax1.imshow(expressions,
                        aspect='auto',
                        cmap='RdBu_r',
                        vmin=-3,
                        vmax=3)
        ax1.set_title('Expression Sequence Heatmap')
        ax1.set_xlabel('Expression Dimension')
        ax1.set_ylabel('Frame')
        plt.colorbar(im1, ax=ax1, label='Normalized Expression Value')
        
        # Plot line graph for selected dimensions
        n_dims = min(5, expressions.shape[1])  # Show first 5 dimensions
        for i in range(n_dims):
            ax2.plot(expressions[:, i], label=f'Dim {i}')
        
        ax2.set_title('Expression Values Over Time')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Normalized Expression Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/expression_sequence_{start_idx}_{start_idx+n_frames}.png',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

    def save_expression_stats(self, motion_data, save_dir: str = 'visualizations') -> None:
        """
        Save statistics about expressions.
        
        Args:
            motion_data: Motion data dictionary
            save_dir: Directory to save statistics
        """
        import numpy as np
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect all expressions
        all_expressions = []
        for frame in motion_data['motion']:
            exp = frame["exp"].flatten()
            normalized_exp = (exp - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps)
            all_expressions.append(normalized_exp)
        
        all_expressions = np.stack(all_expressions)
        
        # Calculate statistics
        stats = {
            'mean': all_expressions.mean(axis=0).tolist(),
            'std': all_expressions.std(axis=0).tolist(),
            'min': all_expressions.min(axis=0).tolist(),
            'max': all_expressions.max(axis=0).tolist(),
            'global_mean': float(all_expressions.mean()),
            'global_std': float(all_expressions.std()),
            'global_min': float(all_expressions.min()),
            'global_max': float(all_expressions.max()),
        }
        
        # Save statistics
        with open(f'{save_dir}/expression_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        return stats
        
    def __len__(self):
        return len(self.all_data)
    
    def check_motion_length(self, motion_data):
        seq_len = motion_data["n_frames"]
        logger.debug(f"Checking motion length: {seq_len} frames")
        
        if seq_len > self.coef_total_len + 2:
            logger.debug("Motion length sufficient")
            return motion_data
        else:
            logger.warning(f"Motion length insufficient ({seq_len}), extending sequence")
            exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
            for frame_index in range(motion_data["n_frames"]):
                exp_list.append(motion_data["motion"][frame_index]["exp"])
                t_list.append(motion_data["motion"][frame_index]["t"])
                scale_list.append(motion_data["motion"][frame_index]["scale"])
                pitch_list.append(motion_data["motion"][frame_index]["pitch"])
                yaw_list.append(motion_data["motion"][frame_index]["yaw"])
                roll_list.append(motion_data["motion"][frame_index]["roll"])

            repeat = 0 
            while len(exp_list) < self.coef_total_len + 2:
                exp_list = exp_list * 2
                t_list = t_list * 2
                scale_list = scale_list * 2
                pitch_list = pitch_list * 2
                yaw_list = yaw_list * 2
                roll_list = roll_list * 2
                repeat += 1
            
            logger.debug(f"Sequence extended {repeat} times")
            
            motion_new = {"motion": []}
            for i in range(len(exp_list)):
                motion = {
                    "exp": exp_list[i],
                    "t": t_list[i],
                    "scale": scale_list[i],
                    "pitch": pitch_list[i],
                    "yaw": yaw_list[i],
                    "roll": roll_list[i],
                }
                motion_new["motion"].append(motion)
            motion_new["n_frames"] = len(exp_list)
            motion_new["repeat"] = repeat
            
            logger.debug(f"New motion length: {motion_new['n_frames']} frames")
            return motion_new
    
    def __getitem__(self, index):
        logger.debug(f"Getting item at index: {index}")
        has_valid_audio = False
        attempts = 0
        max_attempts = 5  # Prevent infinite loops
        
        while not has_valid_audio and attempts < max_attempts:
            attempts += 1
            try:
                # read motion
                metadata = self.all_data[index]
                logger.debug(f"Processing audio file: {metadata['audio_name']}")
                
                motion_data = self.motion_data[metadata["audio_name"]]
                motion_data = self.check_motion_length(motion_data)
                
                # crop audio and coef, count start_frame
                seq_len = motion_data["n_frames"]
                if self.crop_strategy == 'random':
                    end = seq_len - self.coef_total_len
                    if end < 0:
                        logger.warning(f"Invalid sequence length for {os.path.basename(metadata['audio_name'])}, n_frames: {seq_len}")
                        has_valid_audio = False 
                        continue
                    start_frame = np.random.randint(0, seq_len - self.coef_total_len)
                    logger.debug(f"Random crop - start_frame: {start_frame}")
                elif self.crop_strategy == 'begin':
                    start_frame = 0
                elif self.crop_strategy == 'end':
                    start_frame = seq_len - self.coef_total_len
                else:
                    logger.error(f"Unknown crop strategy: {self.crop_strategy}")
                    raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
                
                end_frame = start_frame + self.coef_total_len
                logger.debug(f"Frame range: {start_frame} to {end_frame}")

                # Process motion data
                logger.debug("Processing motion features")
                exp, scale, t, pitch, yaw, roll = [], [], [], [], [], []
                for frame_idx in range(motion_data["n_frames"]):
                    exp.append((motion_data['motion'][frame_idx]["exp"].flatten() - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps))
                    scale.append((motion_data['motion'][frame_idx]["scale"].flatten() - self.templete_dict["mean_scale"]) / (self.templete_dict["std_scale"] + self.eps))
                    t.append((motion_data['motion'][frame_idx]["t"].flatten() - self.templete_dict["mean_t"]) / (self.templete_dict["std_t"] + self.eps))
                    pitch.append((motion_data['motion'][frame_idx]["pitch"].flatten() - self.templete_dict["mean_pitch"]) / (self.templete_dict["std_pitch"] + self.eps))
                    yaw.append((motion_data['motion'][frame_idx]["yaw"].flatten() - self.templete_dict["mean_yaw"]) / (self.templete_dict["std_yaw"] + self.eps))
                    roll.append((motion_data['motion'][frame_idx]["roll"].flatten() - self.templete_dict["mean_roll"]) / (self.templete_dict["std_roll"] + self.eps))

                # Process coefficients
                logger.debug("Processing coefficients")
                coef_keys = ["exp", "pose"]
                coef_dict = {k: [] for k in coef_keys}
                audio = []
                
                for frame_idx in range(start_frame, end_frame):
                    for coef_key in coef_keys:
                        if coef_key == "exp":
                            if self.normalize_type == "mix":
                                normalized_exp = (motion_data['motion'][frame_idx]["exp"].flatten() - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps)
                            else:
                                logger.error("Invalid normalization type for exp")
                                raise RuntimeError("error")
                            coef_dict[coef_key].append([normalized_exp, ])
                        elif coef_key == "pose":
                            if self.normalize_type == "mix":
                                pose_data = np.concatenate((
                                    (motion_data['motion'][frame_idx]["scale"].flatten() - self.templete_dict["min_scale"]) / (self.templete_dict["max_scale"] - self.templete_dict["min_scale"] + self.eps),
                                    (motion_data['motion'][frame_idx]["t"].flatten() - self.templete_dict["min_t"]) / (self.templete_dict["max_t"] - self.templete_dict["min_t"] + self.eps),
                                    (motion_data['motion'][frame_idx]["pitch"].flatten() - self.templete_dict["min_pitch"]) / (self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"] + self.eps),
                                    (motion_data['motion'][frame_idx]["yaw"].flatten() - self.templete_dict["min_yaw"]) / (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"] + self.eps),
                                    (motion_data['motion'][frame_idx]["roll"].flatten() - self.templete_dict["min_roll"]) / (self.templete_dict["max_roll"] - self.templete_dict["min_roll"] + self.eps),
                                ))
                            else:
                                logger.error("Invalid normalization type for pose")
                                raise RuntimeError("pose data error")

                            coef_dict[coef_key].append([pose_data, ])
                        else:
                            logger.error(f"Invalid coefficient key: {coef_key}")
                            raise RuntimeError(f"coef_key error: {coef_key}")
                        
                coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
                
                if coef_dict['exp'].shape[0] != self.coef_total_len:
                    logger.error(f"Invalid coefficient length: {coef_dict['exp'].shape[0]}")
                    raise AssertionError(f'Invalid coef length: {coef_dict["exp"].shape[0]}')

                # Load and process audio
                logger.debug(f"Loading audio from: {metadata['audio_name']}")
                audio_path = metadata["audio_name"]
                audio_clip, sr = torchaudio.load(audio_path)
                audio_clip = audio_clip.squeeze()
                
                if "repeat" in motion_data:
                    logger.debug(f"Repeating audio {motion_data['repeat']} times")
                    for _ in range(motion_data["repeat"]):
                        audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

                if sr != 16000:
                    logger.error(f"Invalid sampling rate: {sr}")
                    raise AssertionError(f'Invalid sampling rate: {sr}')
                
                audio.append(audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)])
                audio = torch.cat(audio, dim=0)
                
                if not (audio.shape[0] == self.coef_total_len * self.audio_unit):
                    logger.warning(f"Invalid audio length: {audio.shape[0]}, expected: {self.coef_total_len * self.audio_unit}")
                    has_valid_audio = False 
                    continue

                # Extract final pairs
                logger.debug("Extracting final pairs")
                keys = ['exp', 'pose']
                audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
                coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},
                            {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]
                
                has_valid_audio = True
                logger.debug("Successfully processed item")


                  # After processing motion data
                if hasattr(self, 'save_visualizations') and self.save_visualizations:
                    self.visualize_expression(exp[0], f"Expression Frame {index}")
                    self.visualize_expression_sequence(motion_data, start_frame, min(20, end_frame-start_frame))
                    stats = self.save_expression_stats(motion_data)
                    logger.debug(f"Expression stats for index {index}: Mean={stats['global_mean']:.3f}, Std={stats['global_std']:.3f}")
            
            
                return audio_pair, coef_pair
                
            except Exception as e:
                logger.error(f"Error processing item {index}: {str(e)}", exc_info=True)
                has_valid_audio = False
                
        if not has_valid_audio:
            logger.error(f"Failed to get valid item after {max_attempts} attempts")
            raise RuntimeError(f"Failed to get valid item at index {index}")


if __name__ == "__main__":
    logger.info("Starting main execution")
    
    data_root = "/media/oem/12TB/JoyVASA/data"
    metainfo_filename = "labels.json"
    motion_filename = "motions.pkl"
    motion_templete_filename = "motion_templete.pkl"

    normalize_type = "mix" 

    try:
        train_dataset = TalkingHeadDatasetHungry(
            data_root, 
            motion_filename=motion_filename, 
            motion_templete_filename=motion_templete_filename,
            split="train", 
            coef_fps=25, 
            n_motions=100, 
            crop_strategy="random", 
            normalize_type=normalize_type
        )
        
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=10, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True
        )
        
        logger.info("Starting data loading loop")
        for batch_idx, (audio_pair, coef_pair) in enumerate(train_loader):
            logger.info(f"Processing batch {batch_idx}")
            logger.debug(f"Audio shapes: {audio_pair[0].shape}, {audio_pair[1].shape}")
            logger.debug(f"Exp shapes: {coef_pair[0]['exp'].shape}, {coef_pair[1]['exp'].shape}")
            logger.debug(f"Pose shapes: {coef_pair[0]['pose'].shape}, {coef_pair[1]['pose'].shape}")
            
            print(f"Batch {batch_idx} - Audio: {audio_pair[0].shape}, {audio_pair[1].shape}, "
                  f"Exp: {coef_pair[0]['exp'].shape}, {coef_pair[1]['exp'].shape}, "
                  f"Pose: {coef_pair[0]['pose'].shape}, {coef_pair[1]['pose'].shape}")
            
    except Exception as e:
        logger.error("Error in main execution", exc_info=True)
        raise