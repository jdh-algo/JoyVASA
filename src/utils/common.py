from functools import reduce
from pathlib import Path

import torch
import torch.nn.functional as F
from logger import logger
import traceback

class NullableArgs:
    def __init__(self, namespace):
        for key, value in namespace.__dict__.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        # when an attribute lookup has not found the attribute
        if key == 'align_mask_width':
            if 'use_alignment_mask' in self.__dict__:
                return 1 if self.use_alignment_mask else 0
            else:
                return 0
        if key == 'no_head_pose':
            return not self.predict_head_pose
        if key == 'no_use_learnable_pe':
            return not self.use_learnable_pe

        return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_option_text(args, parser):
    message = ''
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f'\t[default: {str(default)}]'
        message += f'{str(k):>30}: {str(v):<30}{comment}\n'
    return message


def get_model_path(exp_name, iteration, model_type='DPT'):
    exp_root_dir = Path(__file__).parent.parent / 'experiments' / model_type
    exp_dir = exp_root_dir / exp_name
    if not exp_dir.exists():
        exp_dir = next(exp_root_dir.glob(f'{exp_name}*'))
    model_path = exp_dir / f'checkpoints/iter_{iteration:07}.pt'
    return model_path, exp_dir.relative_to(exp_root_dir)


def get_pose_input(coef_dict, rot_repr, with_global_pose):
    """Get pose input based on rotation representation"""
    logger.debug("\n=== Getting Pose Input ===")
    logger.debug(f"Rotation representation: {rot_repr}")
    logger.debug(f"With global pose: {with_global_pose}")
    
    try:
        if rot_repr == 'aa':
            pose_input = coef_dict['pose'] if with_global_pose else coef_dict['pose'][..., 63:70]
            logger.debug(f"Pose input shape: {pose_input.shape}")
            return pose_input
        else:
            raise ValueError(f'Unknown rotation representation: {rot_repr}')
            
    except Exception as e:
        logger.error(f"Error in get_pose_input: {str(e)}")
        logger.error(traceback.format_exc())
        raise



def get_motion_coef(coef_dict, rot_repr, with_global_pose=False, norm_stats=None):
    """Get combined motion coefficients"""
    logger.debug("\n=== Getting Motion Coefficients ===")
    logger.debug(f"Rotation representation: {rot_repr}")
    logger.debug(f"With global pose: {with_global_pose}")
    logger.debug(f"Using normalization stats: {norm_stats is not None}")

    try:
        if norm_stats is not None:
            logger.debug("Applying normalization stats")
            if rot_repr == 'aa':
                keys = ['exp', 'pose']
                logger.debug(f"Using keys for aa: {keys}")
                
                # Normalize coefficients
                coef_dict = {
                    k: (coef_dict[k] - norm_stats[f'{k}_mean']) / norm_stats[f'{k}_std'] 
                    for k in keys
                }
                logger.debug("Coefficients normalized")
                
                # Get pose input
                pose_coef = get_pose_input(coef_dict, rot_repr, with_global_pose)
                logger.debug(f"Pose coefficients shape: {pose_coef.shape}")
                
                combined = torch.cat([coef_dict['exp'], pose_coef], dim=-1)
                logger.debug(f"Combined coefficients shape: {combined.shape}")
                return combined
                
            elif rot_repr == 'emo':
                logger.debug(f"Input coef_dict keys: {coef_dict.keys()}")
                keys = ['exp', 'pose', 'emotion']
                combined = torch.cat([coef_dict['exp'], coef_dict["pose"], coef_dict["emotion"]], dim=-1)
                logger.debug(f"Combined EMO coefficients shape: {combined.shape}")
                return combined
            else:
                raise ValueError(f'Unknown rotation representation {rot_repr}!')
        else:
            logger.debug("No normalization stats provided")
            if rot_repr == 'aa':
                keys = ['exp', 'pose']
                pose_coef = get_pose_input(coef_dict, rot_repr, with_global_pose)
                logger.debug(f"Pose coefficients shape: {pose_coef.shape}")
                combined = torch.cat([coef_dict['exp'], pose_coef], dim=-1)
                logger.debug(f"Combined coefficients shape: {combined.shape}")
                return combined
            elif rot_repr == 'emo':
                logger.debug(f"Input coef_dict keys: {coef_dict.keys()}")
                keys = ['exp', 'pose', 'emotion']
                combined = torch.cat([coef_dict['exp'], coef_dict["pose"], coef_dict["emotion"]], dim=-1)
                logger.debug(f"Combined EMO coefficients shape: {combined.shape}")
                return combined
            else:
                raise ValueError(f'Unknown rotation representation {rot_repr}!')

    except Exception as e:
        logger.error(f"Error in get_motion_coef: {str(e)}")
        logger.error(traceback.format_exc())
        raise




def get_coef_dict(motion_coef, shape_coef=None, denorm_stats=None, with_global_pose=False, rot_repr='aa'):
    """Convert motion coefficients back to dictionary format"""
    logger.debug("\n=== Converting Motion Coefficients to Dictionary ===")
    logger.debug(f"Motion coefficients shape: {motion_coef.shape}")
    logger.debug(f"Rotation representation: {rot_repr}")
    logger.debug(f"With global pose: {with_global_pose}")
    
    try:
        # Get expression coefficients
        coef_dict = {
            'exp': motion_coef[..., :63]
        }
        logger.debug(f"Expression coefficients shape: {coef_dict['exp'].shape}")
        
        # Handle pose based on rotation representation
        if rot_repr == 'aa':
            if with_global_pose:
                coef_dict['pose'] = motion_coef[..., 63:]
                logger.debug("Using full pose coefficients")
            else:
                placeholder = torch.zeros_like(motion_coef[..., :3])
                coef_dict['pose'] = torch.cat([placeholder, motion_coef[..., -1:]], dim=-1)
                logger.debug("Created pose coefficients with placeholder")
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')
            
        # Apply denormalization if stats provided
        if denorm_stats is not None:
            logger.debug("Applying denormalization stats")
            coef_dict = {
                k: coef_dict[k] * denorm_stats[f'{k}_std'] + denorm_stats[f'{k}_mean'] 
                for k in coef_dict
            }
            
        # Zero out global pose if not using it
        if not with_global_pose:
            if rot_repr == 'aa':
                coef_dict['pose'][..., :3] = 0
                logger.debug("Zeroed out global pose")
            else:
                raise ValueError(f'Unknown rotation representation {rot_repr}!')
                
        logger.debug("Final coefficient dictionary shapes:")
        for k, v in coef_dict.items():
            logger.debug(f"  {k}: {v.shape}")
            
        return coef_dict
        
    except Exception as e:
        logger.error(f"Error in get_coef_dict: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def coef_dict_to_vertices(coef_dict, flame, rot_repr='aa', ignore_global_rot=False, flame_batch_size=512):
    shape = coef_dict['exp'].shape[:-1]
    coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
    n_samples = reduce(lambda x, y: x * y, shape, 1)

    # Convert to vertices
    vert_list = []
    for i in range(0, n_samples, flame_batch_size):
        batch_coef_dict = {k: v[i:i + flame_batch_size] for k, v in coef_dict.items()}
        if rot_repr == 'aa':
            vert, _, _ = flame(
                batch_coef_dict['shape'], batch_coef_dict['exp'], batch_coef_dict['pose'],
                pose2rot=True, ignore_global_rot=ignore_global_rot, return_lm2d=False, return_lm3d=False)
        else:
            raise ValueError(f'Unknown rot_repr: {rot_repr}')
        vert_list.append(vert)

    vert_list = torch.cat(vert_list, dim=0)  # (n_samples, 5023, 3)
    vert_list = vert_list.view(*shape, -1, 3)  # (..., 5023, 3)

    return vert_list

def _truncate_audio(audio, end_idx, pad_mode='zero'):
    """Truncate audio and pad remainder"""
    logger.debug("\n=== Truncating Audio ===")
    logger.debug(f"Audio input shape: {audio.shape}")
    logger.debug(f"End indices: {end_idx}")
    logger.debug(f"Pad mode: {pad_mode}")
    
    try:
        batch_size = audio.shape[0]
        audio_trunc = audio.clone()
        
        if pad_mode == 'replicate':
            logger.debug("Using replicate padding")
            for i in range(batch_size):
                audio_trunc[i, end_idx[i]:] = audio_trunc[i, end_idx[i] - 1]
        elif pad_mode == 'zero':
            logger.debug("Using zero padding")
            for i in range(batch_size):
                audio_trunc[i, end_idx[i]:] = 0
        else:
            raise ValueError(f'Unknown pad mode {pad_mode}!')
            
        logger.debug(f"Truncated audio shape: {audio_trunc.shape}")
        return audio_trunc
        
    except Exception as e:
        logger.error(f"Error in _truncate_audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def _truncate_coef_dict(coef_dict, end_idx, pad_mode='zero'):
    """Truncate coefficient dictionary and pad remainder"""
    logger.debug("\n=== Truncating Coefficient Dictionary ===")
    logger.debug(f"End indices: {end_idx}")
    logger.debug(f"Pad mode: {pad_mode}")
    logger.debug("Input coefficient shapes:")
    for k, v in coef_dict.items():
        logger.debug(f"  {k}: {v.shape}")
    
    try:
        batch_size = coef_dict['exp'].shape[0]
        coef_dict_trunc = {k: v.clone() for k, v in coef_dict.items()}
        
        if pad_mode == 'replicate':
            logger.debug("Using replicate padding")
            for i in range(batch_size):
                for k in coef_dict_trunc:
                    coef_dict_trunc[k][i, end_idx[i]:] = coef_dict_trunc[k][i, end_idx[i] - 1]
        elif pad_mode == 'zero':
            logger.debug("Using zero padding")
            for i in range(batch_size):
                for k in coef_dict:
                    coef_dict_trunc[k][i, end_idx[i]:] = 0
        else:
            raise ValueError(f'Unknown pad mode: {pad_mode}!')
            
        logger.debug("Truncated coefficient shapes:")
        for k, v in coef_dict_trunc.items():
            logger.debug(f"  {k}: {v.shape}")
            
        return coef_dict_trunc
        
    except Exception as e:
        logger.error(f"Error in _truncate_coef_dict: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def truncate_motion_coef_and_audio(audio, motion_coef, n_motions, audio_unit=640, pad_mode='zero'):
    """
    Truncate motion coefficients and corresponding audio at random points.
    
    Args:
        audio: Audio tensor [B, T]
        motion_coef: Motion coefficients [B, T, D]
        n_motions: Number of motion frames
        audio_unit: Audio samples per motion frame
        pad_mode: Padding mode ('zero' or 'replicate')
        
    Returns:
        audio_trunc: Truncated audio
        motion_coef_trunc: Truncated motion coefficients
        end_idx: Truncation indices
    """
    logger.debug("\n=== Starting Motion & Audio Truncation ===")
    logger.debug(f"Initial shapes:")
    logger.debug(f"  Audio: {audio.shape}")
    logger.debug(f"  Motion coefficients: {motion_coef.shape}")
    logger.debug(f"Parameters:")
    logger.debug(f"  n_motions: {n_motions}")
    logger.debug(f"  audio_unit: {audio_unit}")
    logger.debug(f"  pad_mode: {pad_mode}")
    
    try:
        # Generate random truncation points
        batch_size = audio.shape[0]
        end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
        audio_end_idx = (end_idx * audio_unit).long()
        
        logger.debug(f"Generated indices:")
        logger.debug(f"  Motion end indices: {end_idx}")
        logger.debug(f"  Audio end indices: {audio_end_idx}")
        
        # Truncate audio
        logger.debug("\nTruncating audio...")
        audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)
        logger.debug(f"Truncated audio shape: {audio_trunc.shape}")
        
        # Prepare coefficient dictionary
        logger.debug("\nPreparing coefficient dictionary...")
        coef_dict = {
            'exp': motion_coef[..., :63],
            'pose_any': motion_coef[..., 63:]
        }
        logger.debug("Split motion coefficients:")
        logger.debug(f"  Expression shape: {coef_dict['exp'].shape}")
        logger.debug(f"  Pose shape: {coef_dict['pose_any'].shape}")
        
        # Truncate coefficients
        logger.debug("\nTruncating coefficients...")
        coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)
        
        # Recombine truncated coefficients
        motion_coef_trunc = torch.cat([
            coef_dict_trunc['exp'], 
            coef_dict_trunc['pose_any']
        ], dim=-1)
        
        logger.debug("\nFinal outputs:")
        logger.debug(f"  Truncated audio shape: {audio_trunc.shape}")
        logger.debug(f"  Truncated motion shape: {motion_coef_trunc.shape}")
        logger.debug(f"  End indices shape: {end_idx.shape}")
        
        return audio_trunc, motion_coef_trunc, end_idx
        
    except Exception as e:
        logger.error(f"Error in truncate_motion_coef_and_audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def nt_xent_loss(feature_a, feature_b, temperature):
    """
    Normalized temperature-scaled cross entropy loss.

    (Adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py)

    Args:
        feature_a (torch.Tensor): shape (batch_size, feature_dim)
        feature_b (torch.Tensor): shape (batch_size, feature_dim)
        temperature (float): temperature scaling factor

    Returns:
        torch.Tensor: scalar
    """
    batch_size = feature_a.shape[0]
    device = feature_a.device

    features = torch.cat([feature_a, feature_b], dim=0)

    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)

    # select the positives and negatives
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    labels = torch.zeros(labels.shape[0], dtype=torch.long).to(device)

    loss = F.cross_entropy(logits, labels)
    return loss


def compute_loss_new(args, is_starting_sample, motion_coef_gt, noise, target, prev_motion_coef, end_idx=None):
    if args.criterion.lower() == 'l2':
        criterion_func = F.mse_loss
    elif args.criterion.lower() == 'l1':
        criterion_func = F.l1_loss
    else:
        raise NotImplementedError(f'Criterion {args.criterion} not implemented.')

    # 表情类损失
    loss_exp = None
    loss_exp_vel = None
    loss_exp_smooth = None
    # 头部运动类
    loss_head_angle = None
    loss_head_vel = None
    loss_head_smooth = None
    # Trans类
    loss_head_trans_vel = None
    loss_head_trans_accel = None
    loss_head_trans = None
    if args.target == 'noise':
        loss_noise = criterion_func(noise, target[:, args.n_prev_motions:], reduction='none')
    elif args.target == 'sample':
        if is_starting_sample:
            target = target[:, args.n_prev_motions:]
        else:
            motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_gt], dim=1)
            if args.no_constrain_prev:
                target = torch.cat([prev_motion_coef, target[:, args.n_prev_motions:]], dim=1)

        # print(f"loss, motion_coef_gt: {motion_coef_gt.shape}, target: {target.shape}")
        loss_noise = criterion_func(motion_coef_gt, target, reduction='none')
        # print("loss_noise: ", loss_noise)

        # 表情相关损失
        if args.rot_repr == "aa":
            exp_gt = motion_coef_gt[:, :, :63]
            exp_pred = target[:, :, :63]
        elif args.rot_repr == "emo":
            exp_gt = torch.cat([motion_coef_gt[:, :, :63], motion_coef_gt[:, :, -3:]], -1)
            exp_pred = torch.cat([target[:, :, :63], target[:, :, -3:]], -1)
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

        loss_exp = criterion_func(exp_gt, exp_pred, reduction='none')
        if args.l_exp_vel > 0:
            vel_exp_gt = exp_gt[:, 1:] - exp_gt[:, :-1]
            vel_exp_pred = exp_pred[:, 1:] - exp_pred[:, :-1]
            loss_exp_vel = criterion_func(vel_exp_gt, vel_exp_pred, reduction='none')
        if args.l_exp_smooth > 0:
            vel_exp_pred = exp_pred[:, 1:] - exp_pred[:, :-1]
            loss_exp_smooth = criterion_func(vel_exp_pred[:, 1:], vel_exp_pred[:, :-1], reduction='none')

        # 头部运动相关损失
        if not args.no_head_pose:
            if args.rot_repr == 'aa': # 旋转表征，aa，单独计算exp和（R, t, delta）
                head_pose_gt = motion_coef_gt[:, :, 63:]
                head_pose_pred = target[:, :, 63:]
            elif args.rot_repr == 'emo': # 旋转表征，aa，单独计算exp和（R, t, delta）
                head_pose_gt = motion_coef_gt[:, :, 63:70]
                head_pose_pred = target[:, :, 63:70]
            else:
                raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

            # angle, gt_pose和pred_pose之间的损失
            if args.l_head_angle > 0:
                loss_head_angle = criterion_func(head_pose_gt, head_pose_pred, reduction='none')
            if args.l_head_vel > 0:
                # print("head_pose_gt: ", head_pose_gt.shape, head_pose_pred.shape)
                head_vel_gt = head_pose_gt[:, 1:] - head_pose_gt[:, :-1]
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                loss_head_vel = criterion_func(head_vel_gt, head_vel_pred, reduction='none')
            if args.l_head_smooth > 0:
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                loss_head_smooth = criterion_func(head_vel_pred[:, 1:], head_vel_pred[:, :-1], reduction='none')

            if not is_starting_sample and args.l_head_trans > 0:
                # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})
                # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})
                head_pose_trans = torch.cat([head_pose_gt[:, args.n_prev_motions - 3:args.n_prev_motions],
                                             head_pose_pred[:, args.n_prev_motions:args.n_prev_motions + 3]], dim=1)
                head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]
                # will constrain x_{-2|0} ~ x_{1}
                loss_head_trans_vel = criterion_func(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3], reduction='none')
                # will constrain x_{-3|0} ~ x_{2}
                loss_head_trans_accel = criterion_func(head_accel_pred[:, 1:], head_accel_pred[:, :-1], reduction='none')
    else:
        raise ValueError(f'Unknown diffusion target: {args.target}')

    if end_idx is None:
        mask = torch.ones((target.shape[0], args.n_motions), dtype=torch.bool, device=target.device)
    else:
        mask = torch.arange(args.n_motions, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)

    if args.target == 'sample' and not is_starting_sample:
        if args.no_constrain_prev:
            # Warning: this option will be deprecated in the future
            mask = torch.cat([torch.zeros_like(mask[:, :args.n_prev_motions]), mask], dim=1)
        else:
            mask = torch.cat([torch.ones_like(mask[:, :args.n_prev_motions]), mask], dim=1)

    loss_noise = loss_noise[mask].mean()
    if loss_exp is not None:
        loss_exp = loss_exp[mask].mean()
    if loss_exp_vel is not None:
        loss_exp_vel = loss_exp_vel[mask[:, 1:]].mean()
    if loss_exp_smooth is not None:
        loss_exp_smooth = loss_exp_smooth[mask[:, 2:]].mean()
    if loss_head_angle is not None:
        loss_head_angle = loss_head_angle[mask].mean()
    if loss_head_vel is not None:
        loss_head_vel = loss_head_vel[mask[:, 1:]]
        loss_head_vel = loss_head_vel.mean() if torch.numel(loss_head_vel) > 0 else None
    if loss_head_smooth is not None:
        loss_head_smooth = loss_head_smooth[mask[:, 2:]]
        loss_head_smooth = loss_head_smooth.mean() if torch.numel(loss_head_smooth) > 0 else None
    if loss_head_trans_vel is not None:
        vel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 2]
        accel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 3]
        loss_head_trans_vel = loss_head_trans_vel[vel_mask].mean()
        loss_head_trans_accel = loss_head_trans_accel[accel_mask].mean()
        loss_head_trans = loss_head_trans_vel + loss_head_trans_accel

    return loss_noise, loss_exp, loss_exp_vel, loss_exp_smooth, loss_head_angle, loss_head_vel, loss_head_smooth, loss_head_trans