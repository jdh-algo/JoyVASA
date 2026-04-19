import os
import tyro
import multiprocessing
import sys
import subprocess
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath("../")))

from src.config.argument_config import ArgumentConfig
from src.motion_extractor import make_motion_templete
from logger import logger

def extract_audio(video_path):
    """Extract audio from video file as WAV with 16kHz sampling rate"""
    wav_path = str(Path(video_path).with_suffix('.wav'))
    if os.path.exists(wav_path):
        print(f"Audio already extracted: {wav_path}")
        return wav_path
        
    try:
        # First extract at original quality
        temp_wav = str(Path(video_path).with_suffix('.temp.wav'))
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Use PCM codec
            '-y',  # Overwrite output file
            temp_wav
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(f"FFmpeg output: {result.stderr}")

        # Then resample to 16kHz
        command = [
            'ffmpeg',
            '-i', temp_wav,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            '-y',           # Overwrite output file
            wav_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(f"FFmpeg resampling output: {result.stderr}")

        # Clean up temporary file
        os.remove(temp_wav)
        print(f"Extracted and resampled audio to: {wav_path}")
        return wav_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error processing {video_path}: {e.stderr}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise
    except Exception as e:
        logger.error(f"Unexpected error extracting audio from {video_path}: {str(e)}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise

def process_video(args, video_path, suffix):
    """Process a single video - extract motion template and audio"""
    try:
        # Extract motion template
        make_motion_templete(args, video_path, suffix)
        
        # Extract audio
        extract_audio(video_path)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def process_videos(args, video_list, suffix):
    """Process multiple videos in parallel"""
    params = [(args, driving_video, suffix) for driving_video in video_list]
    
    # Use half of available CPU cores, but minimum of 1
    n_processes = max(1, multiprocessing.cpu_count() // 2)
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.starmap(process_video, params)

def main():
    args = tyro.cli(ArgumentConfig)
    args.flag_do_crop = False
    args.scale = 2.3

    root_dir = "/media/oem/12TB/JoyVASA/data"
    video_names = sorted([
        os.path.join(root_dir, filename) 
        for filename in os.listdir(root_dir) 
        if filename.endswith(".mp4")
    ])
    
    process_videos(args, video_names, suffix=".pkl")

if __name__ == "__main__":
    main()
