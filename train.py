import argparse
from collections import deque, defaultdict
from pathlib import Path

import os
import sys
import logging
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

import src.utils as utils
from src.dataset import infinite_data_loader
from src.dataset.talkinghead_dataset_hungry import TalkingHeadDatasetHungry
from src.modules.dit_talking_head import DitTalkingHead
from logger import logger
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import math
import wandb
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from collections import defaultdict



def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, ):
    save_dir.mkdir(parents=True, exist_ok=True)

    # model
    device = model.device
    model.train()

    # data
    data_loader = infinite_data_loader(train_loader)
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))

    optimizer.zero_grad()
    for it in range(args.max_iter + 1):
        # Load data
        audio_pair, coef_pair = next(data_loader)
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
        motion_coef_pair = [
            utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
        ] 

        # Extract audio features
        if args.use_context_audio_feat:
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

        loss_noise = 0
        loss_exp = torch.tensor(0, device=device)
        loss_exp_v = torch.tensor(0, device=device)
        loss_exp_s = torch.tensor(0, device=device)
        loss_head_angle = torch.tensor(0, device=device)
        loss_head_vel = torch.tensor(0, device=device)
        loss_head_smooth = torch.tensor(0, device=device)
        loss_head_trans = 0
        for i in range(2):
            audio = audio_pair[i]  # (N, L_a)
            motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
            batch_size = audio.shape[0]

            # truncate input audio and motion according to trunc_prob
            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                if args.use_context_audio_feat and i != 0:
                    # use contextualized audio feature for the second clip
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                           args.n_motions * 2)[:, -args.n_motions:]
            else:
                if args.use_context_audio_feat:
                    audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                else:
                    audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            if args.use_indicator:
                if end_idx is not None:
                        # Log the creation process
                        arange_tensor = torch.arange(args.n_motions, device=device)
                        logger.debug(f"Arange tensor: {arange_tensor}")
                        
                        expanded = arange_tensor.expand(batch_size, -1)
                        logger.debug(f"Expanded shape: {expanded.shape}")
                        logger.debug(f"First row of expanded: {expanded[0]}")
                        
                        comparison = expanded < end_idx.unsqueeze(1)
                        logger.debug(f"end_idx unsqueezed: {end_idx.unsqueeze(1).shape}")
                        
                        indicator = comparison
                        logger.debug(f"Indicator shape: {indicator.shape}")
                        logger.debug(f"First row indicator sums: {indicator[0].sum()}")
                        logger.debug(f"Indicator example:\n{indicator[0]}")  # Show first batch item
                    
                  
                        
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = None

            # Inference
            if i == 0:
                noise, target, prev_motion_coef, prev_audio_feat = model(motion_coef_in, audio_in, indicator=indicator)
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                noise, target, _, _ = model(motion_coef_in, audio_in, prev_motion_coef, prev_audio_feat, indicator=indicator)

            if it % 50 == 0:
                visualize_expression_comparison(target, noise, iteration=it)
            # visualize_expression_sequence_comparison(target, noise, n_frames=5, iteration=it)
      
            loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)
            loss_noise = loss_noise + loss_n / 2
            loss_exp = loss_exp + loss_exp / 2
            loss_exp_v = loss_exp_v + loss_exp_v / 2.
            loss_exp_s = loss_exp_s + loss_exp_s / 2.
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_head_angle = loss_head_angle + loss_ha / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                loss_head_vel = loss_head_vel + loss_hc / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                # no need to divide by 2 because it only applies to the second clip
                loss_head_trans = loss_head_trans + loss_ht

        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise

        loss_log['exp'].append(loss_exp.item() * args.l_exp)
        loss = loss + args.l_exp * loss_exp

        loss_log['exp_vel'].append(loss_exp_v.item() * args.l_exp_vel)
        loss = loss + args.l_exp_vel * loss_exp_v

        loss_log['exp_smooth'].append(loss_exp_s.item() * args.l_exp_smooth)
        loss = loss + args.l_exp_smooth * loss_exp_s

        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            loss_log['head_angle'].append(loss_head_angle.item() * args.l_head_angle)
            loss = loss + args.l_head_angle * loss_head_angle

        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            loss_log['head_vel'].append(loss_head_vel.item() * args.l_head_vel)
            loss = loss + args.l_head_vel * loss_head_vel

        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            loss_log['head_smooth'].append(loss_head_smooth.item() * args.l_head_smooth)
            loss = loss + args.l_head_smooth * loss_head_smooth
        
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            loss_log['head_trans'].append(loss_head_trans.item() * args.l_head_trans)
            loss = loss + args.l_head_trans * loss_head_trans

        loss.backward()

        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        loss_log['loss'].append(loss.item())
        description = f'Iter: {it}\t  Train loss: [N: {np.mean(loss_log["noise"]):.3e}'
        description += f", EX: {np.mean(loss_log['exp']):.3e}"
        description += f", EX_V: {np.mean(loss_log['exp_vel']):.3e}"
        description += f", EX_S: {np.mean(loss_log['exp_smooth']):.3e}"
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            description += f', HV: {np.mean(loss_log["head_vel"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
        description += ']'
        logging.info(description)

        # write to tensorboard
        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/simple_loss', np.mean(loss_log['noise']), it)
            writer.add_scalar('train/exp_loss', np.mean(loss_log['exp']), it)
            writer.add_scalar('train/exp_vel_loss', np.mean(loss_log['exp_vel']), it)
            writer.add_scalar('train/exp_smooth_loss', np.mean(loss_log['exp_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                writer.add_scalar('train/head_angle', np.mean(loss_log['head_angle']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                writer.add_scalar('train/head_vel', np.mean(loss_log['head_vel']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                writer.add_scalar('train/head_smooth', np.mean(loss_log['head_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                writer.add_scalar('train/head_trans', np.mean(loss_log['head_trans']), it)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)

        # update learning rate
        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        # save model
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save({
                'args': args,
                'model': model.state_dict(),
                'iter': it,
            }, save_dir / f'iter_{it:07}.pt')

        # validation
        if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:
            val(args, model, val_loader, it, 1, 'val', writer)

def visualize_expression_comparison(target, noise, save_dir='visualizations/expressions', iteration=None):
    """
    Visualize comparison between target and predicted expressions.
    
    Args:
        target: Target expression tensor
        noise: Predicted noise tensor
        save_dir: Directory to save visualizations
        iteration: Current training iteration (for filename)
    """
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    import torch
    import numpy as np

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays and move to CPU if needed
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(noise):
        noise = noise.detach().cpu().numpy()

    # Take first sample from batch and first frame
    target_frame = target[0, 0]  # First batch item, first frame
    noise_frame = noise[0, 0]    # First batch item, first frame
    
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot target expression
    im1 = ax1.imshow(target_frame.reshape(1, -1), 
                     aspect='auto', 
                     cmap='RdBu_r',
                     vmin=-3, 
                     vmax=3)
    ax1.set_title('Target Expression')
    plt.colorbar(im1, ax=ax1)
    ax1.set_yticks([])

    # Plot predicted expression
    im2 = ax2.imshow(noise_frame.reshape(1, -1), 
                     aspect='auto', 
                     cmap='RdBu_r',
                     vmin=-3, 
                     vmax=3)
    ax2.set_title('Predicted Expression')
    plt.colorbar(im2, ax=ax2)
    ax2.set_yticks([])

    # Plot difference
    difference = noise_frame - target_frame
    max_diff = np.abs(difference).max()
    im3 = ax3.imshow(difference.reshape(1, -1),
                     aspect='auto',
                     cmap='RdBu_r',
                     vmin=-max_diff,
                     vmax=max_diff)
    ax3.set_title('Difference (Predicted - Target)')
    plt.colorbar(im3, ax=ax3)
    ax3.set_yticks([])

    # Add titles and adjust layout
    if iteration is not None:
        plt.suptitle(f'Expression Comparison - Iteration {iteration}')
    plt.tight_layout()

    # Save the figure
    filename = f'expression_comparison_{iteration if iteration else "latest"}.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def visualize_expression_sequence_comparison(target, noise, n_frames=5, save_dir='visualizations/expressions', iteration=None):
    """
    Visualize sequence of target vs predicted expressions.
    
    Args:
        target: Target expression tensor (batch_size, seq_len, feat_dim)
        noise: Predicted noise tensor (batch_size, seq_len, feat_dim)
        n_frames: Number of frames to visualize
        save_dir: Directory to save visualizations
        iteration: Current training iteration (for filename)
    """
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    import torch
    import numpy as np

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays and move to CPU if needed
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(noise):
        noise = noise.detach().cpu().numpy()

    # Take first sample from batch
    target = target[0]  # First batch item
    noise = noise[0]    # First batch item

    # Get actual sequence length (minimum of both sequences)
    seq_len = min(target.shape[0], noise.shape[0])
    n_frames = min(n_frames, seq_len)  # Ensure we don't exceed sequence length
    
    # Select frames evenly spaced through the sequence
    frame_indices = np.linspace(0, seq_len-1, n_frames, dtype=int)

    # Create subplot grid
    fig, axes = plt.subplots(n_frames, 3, figsize=(15, 3*n_frames))
    
    # If only one frame, wrap axes in list for consistent indexing
    if n_frames == 1:
        axes = axes.reshape(1, -1)

    # For storing global min/max values
    global_min = min(target.min(), noise.min())
    global_max = max(target.max(), noise.max())
    
    for idx, frame_idx in enumerate(frame_indices):
        # Ensure we don't exceed array bounds
        safe_idx = min(frame_idx, seq_len-1)
        
        # Plot target
        im1 = axes[idx, 0].imshow(target[safe_idx].reshape(1, -1), 
                                 aspect='auto', 
                                 cmap='RdBu_r',
                                 vmin=global_min, 
                                 vmax=global_max)
        axes[idx, 0].set_title(f'Target (Frame {safe_idx})' if idx == 0 else f'Frame {safe_idx}')
        axes[idx, 0].set_yticks([])

        # Plot prediction
        im2 = axes[idx, 1].imshow(noise[safe_idx].reshape(1, -1), 
                                 aspect='auto', 
                                 cmap='RdBu_r',
                                 vmin=global_min, 
                                 vmax=global_max)
        axes[idx, 1].set_title('Prediction' if idx == 0 else '')
        axes[idx, 1].set_yticks([])

        # Plot difference
        difference = noise[safe_idx] - target[safe_idx]
        max_diff = np.abs(difference).max()
        im3 = axes[idx, 2].imshow(difference.reshape(1, -1), 
                                 aspect='auto', 
                                 cmap='RdBu_r',
                                 vmin=-max_diff, 
                                 vmax=max_diff)
        axes[idx, 2].set_title('Difference' if idx == 0 else '')
        axes[idx, 2].set_yticks([])

    # Add colorbars
    plt.colorbar(im1, ax=axes[:, 0], label='Expression Value')
    plt.colorbar(im2, ax=axes[:, 1], label='Expression Value')
    plt.colorbar(im3, ax=axes[:, 2], label='Difference')

    # Add title and adjust layout
    if iteration is not None:
        plt.suptitle(f'Expression Sequence Comparison - Iteration {iteration}')
    plt.tight_layout()

    # Save the figure
    filename = f'expression_sequence_comparison_{iteration if iteration else "latest"}.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    
@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode='val', writer=None):
    print("test ... ")
    is_training = model.training
    device = model.device
    model.eval()

    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(list)
    for test_round in range(n_rounds):
        for audio_pair, coef_pair in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            motion_coef_pair = [
                utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
            ]  # (N, L, 50+x)

            # Extract audio features
            if args.use_context_audio_feat:
                audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

            loss_noise = 0
            loss_exp = 0
            loss_exp_v = 0
            loss_exp_s = 0
            loss_head_angle = 0
            loss_head_vel = torch.tensor(0, device=device)
            loss_head_smooth = torch.tensor(0, device=device)
            loss_head_trans = 0
            for i in range(2):
                audio = audio_pair[i]  # (N, L_a)
                motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
                batch_size = audio.shape[0]

                # truncate input audio and motion according to trunc_prob
                if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                    audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                    if args.use_context_audio_feat and i != 0:
                        # use contextualized audio feature for the second clip
                        audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                               args.n_motions * 2)[:, -args.n_motions:]
                else:
                    if args.use_context_audio_feat:
                        audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                    else:
                        audio_in = audio
                    motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    if end_idx is not None:
                        indicator = torch.arange(args.n_motions, device=device).expand(batch_size,
                                                                                       -1) < end_idx.unsqueeze(1)
                    else:
                        indicator = torch.ones(batch_size, args.n_motions, device=device)
                else:
                    indicator = None

                # Inference
                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat = model(motion_coef_in, audio_in, indicator=indicator)
                    if end_idx is not None:  # was truncated, needs to use the complete feature
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:
                            prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        else:
                            with torch.no_grad():
                                prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    noise, target, _, _ = model(motion_coef_in, audio_in, prev_motion_coef, prev_audio_feat, indicator=indicator)

                loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)


            

                # simple loss
                loss_noise = loss_noise + loss_n / 2

                # exp-related loss 
                loss_exp = loss_exp + loss_exp / 2
                loss_exp_v = loss_exp_v + loss_exp_v / 2
                loss_exp_s = loss_exp_s + loss_exp_s / 2
                
                # head pose loss 
                if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                    loss_head_angle = loss_head_angle + loss_ha / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                    loss_head_vel = loss_head_vel + loss_hc / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                    loss_head_smooth = loss_head_smooth + loss_hs / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                    # no need to divide by 2 because it only applies to the second clip
                    loss_head_trans = loss_head_trans + loss_ht

            loss_log['noise'].append(loss_noise.item())
            loss = loss_noise
            
            loss_log['exp'].append(loss_exp.item() * args.l_exp)
            loss = loss + args.l_exp * loss_exp

            loss_log['exp_vel'].append(loss_exp_v.item() * args.l_exp_vel)
            loss = loss + args.l_exp_vel * loss_exp_v

            loss_log['exp_smooth'].append(loss_exp_s.item() * args.l_exp_smooth)
            loss = loss + args.l_exp_smooth * loss_exp_s

            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_log['head_angle'].append(loss_head_angle.item() * args.l_head_angle)
                loss = loss + args.l_head_angle * loss_head_angle

            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                loss_log['head_vel'].append(loss_head_vel.item() * args.l_head_vel)
                loss = loss + args.l_head_vel * loss_head_vel

            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                loss_log['head_smooth'].append(loss_head_smooth.item() * args.l_head_smooth)
                loss = loss + args.l_head_smooth * loss_head_smooth

            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                loss_log['head_trans'].append(loss_head_trans.item() * args.l_head_trans)
                loss = loss + args.l_head_trans * loss_head_trans

            loss_log['loss'].append(loss.item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [N: {np.mean(loss_log["noise"]):.3e}'
    description += f", EX: {np.mean(loss_log['exp']):.3e}"
    description += f", EX_V: {np.mean(loss_log['exp_vel']):.3e}"
    description += f", EX_S: {np.mean(loss_log['exp_smooth']):.3e}"
    if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
        description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
        description += f', HV: {np.mean(loss_log["head_vel"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
        description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
        description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
    description += ']'
    print(description)

    # write to tensorboard
    if writer is not None:
        writer.add_scalar(f'{mode}/total_loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/simplt_loss', np.mean(loss_log['noise']), current_iter)
        writer.add_scalar('train/exp_loss', np.mean(loss_log['exp']), current_iter)
        writer.add_scalar('train/exp_vel_loss', np.mean(loss_log['exp_vel']), current_iter)
        writer.add_scalar('train/exp_smooth_loss', np.mean(loss_log['exp_smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            writer.add_scalar(f'{mode}/head_angle', np.mean(loss_log['head_angle']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            writer.add_scalar(f'{mode}/head_vel', np.mean(loss_log['head_vel']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            writer.add_scalar(f'{mode}/head_smooth', np.mean(loss_log['head_smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            writer.add_scalar(f'{mode}/head_trans', np.mean(loss_log['head_trans']), current_iter)

    if is_training:
        model.train()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args, option_text=None):
    # model
    model_kwargs = dict(
        device = device,
        target = args.target,
        architecture = args.architecture,
        motion_feat_dim = args.motion_feat_dim,
        fps = args.fps, 
        n_motions = args.n_motions,
        n_prev_motions = args.n_prev_motions,
        audio_model = args.audio_model,
        feature_dim = args.feature_dim,
        n_diff_steps = args.n_diff_steps,
        diff_schedule = args.diff_schedule,
        cfg_mode = args.cfg_mode,
        guiding_conditions = args.guiding_conditions,
    )
    model = DitTalkingHead(**model_kwargs)

    # Dataset
    train_dataset = TalkingHeadDatasetHungry(args.data_root, motion_filename=args.motion_filename, 
                                             motion_templete_filename=args.motion_templete_filename, split="train", coef_fps=args.fps, n_motions=args.n_motions, 
                                             crop_strategy=args.crop_strategy, normalize_type=args.normalize_type)
    val_dataset = TalkingHeadDatasetHungry(args.data_root, motion_filename=args.motion_filename, 
                                           motion_templete_filename=args.motion_templete_filename, split="val", coef_fps=args.fps, n_motions=args.n_motions, 
                                           crop_strategy=args.crop_strategy, normalize_type=args.normalize_type)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Logging
    exp_dir = Path('experiments/JoyVASA') / f'{args.exp_name}'
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / 'options.log', 'w') as f:
            f.write(option_text)
        writer.add_text('options', option_text)



    # optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.scheduler == 'Warmup':
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == 'WarmupThenDecay':
        from src.scheduler import GradualWarmupScheduler
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_max_iter - args.warm_iter,
                                                                args.lr * args.min_lr_ratio)
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
    else:
        scheduler = None

    # train
    train(args, model, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', scheduler, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--iter', type=int, default=1, help='iteration to test')
    parser.add_argument('--exp_name', type=str, default='test_b16', help='experiment name')

    # Dataset
    parser.add_argument('--data_root', type=Path, default="data/",)
    parser.add_argument('--motion_filename', type=str, default='motions.pkl')
    parser.add_argument('--motion_templete_filename', type=str, default='motion_templete.pkl')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--crop_strategy', type=str, default="random")
    parser.add_argument('--normalize_type', type=str, default="mix", choices=["std", "case", "scale", "minmax", "mix"])

    # Model
    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=50, help='number of diffusion steps')
    parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--no_head_pose', action='store_true', default=False, help='do not predict head pose')
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])

    # transformer
    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=1, help='width of the alignment mask, non-positive for no mask')
    parser.add_argument('--no_use_learnable_pe', action='store_true', help='do not use learnable positional encoding')
    parser.add_argument('--use_indicator', action='store_true', default=True, help='use indicator for padded frames')
    parser.add_argument('--feature_dim', type=int, default=256, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    # sequence
    parser.add_argument('--n_motions', type=int, default=100, help='number of motions in a sequence')
    parser.add_argument('--n_prev_motions', type=int, default=25, help='number of pre-motions in a sequence')
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25, help='frame per second')
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    # Training
    parser.add_argument('--max_iter', type=int, default=50000, help='max number of iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--scheduler', type=str, default='None', choices=['None', 'Warmup', 'WarmupThenDecay'])

    # 损失函数 & 权重
    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--l_exp', type=float, default=0.1, help='weight of the head angle loss')
    parser.add_argument('--l_exp_vel', type=float, default=1e-4, help='weight of the head angle loss')
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4, help='weight of the head angle loss')
    parser.add_argument('--l_head_angle', type=float, default=1e-2, help='weight of the head angle loss')
    parser.add_argument('--l_head_vel', type=float, default=1e-2, help='weight of the head angular velocity loss')
    parser.add_argument('--l_head_smooth', type=float, default=1e-2, help='weight of the head angular acceleration regularization')
    parser.add_argument('--l_head_trans', type=float, default=1e-2, help='weight of the head constraint during window transition')
    parser.add_argument('--no_constrain_prev', action='store_true', help='do not constrain the generated previous motions')

    parser.add_argument('--use_context_audio_feat', action='store_true')
    parser.add_argument('--trunc_prob1', type=float, default=0.3, help='truncation probability for the first sample')
    parser.add_argument('--trunc_prob2', type=float, default=0.4, help='truncation probability for the second sample')

    parser.add_argument('--save_iter', type=int, default=1000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=50, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')

    # warm_up
    parser.add_argument('--warm_iter', type=int, default=2000)
    parser.add_argument('--cos_max_iter', type=int, default=12000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)

    args = parser.parse_args()

    if args.mode == 'train':
        option_text = utils.common.get_option_text(args, parser)
    else:
        option_text = None

    main(args, option_text)