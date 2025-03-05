import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path
from contextlib import contextmanager
from logger import logger
import traceback
import numpy as np
import math

class FunctionalModule(nn.Module):
    """
    Enhanced FunctionalModule that properly handles nested parameters and ensures gradient flow.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.fast_params = {}
        self._create_fast_params(module)
    
    def _create_fast_params(self, module, prefix=''):
        """Recursively create fast parameters for all nested modules."""
        for name, param in module.named_parameters(recurse=False):
            param_name = f"{prefix}.{name}" if prefix else name
            self.fast_params[param_name] = param.clone().detach().requires_grad_(True)
            
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._create_fast_params(child, prefix=child_prefix)
    
    def _update_module_params(self, module, prefix=''):
        """Recursively update module parameters with fast parameters."""
        for name, param in module.named_parameters(recurse=False):
            param_name = f"{prefix}.{name}" if prefix else name
            if param_name in self.fast_params:
                param.data = self.fast_params[param_name].data
                
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._update_module_params(child, prefix=child_prefix)
    
    def forward(self, *args, params=None, **kwargs):
        if params is not None:
            self.fast_params = params
        
        # Temporarily update module parameters with fast parameters
        original_params = {}
        for name, param in self.module.named_parameters():
            original_params[name] = param.data.clone()
            if name in self.fast_params:
                param.data = self.fast_params[name].data
        
        try:
            output = self.module(*args, **kwargs)
        finally:
            # Restore original parameters
            for name, param in self.module.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        
        return output
    
    def parameters(self):
        """Return fast parameters in the same order as module.parameters()."""
        for name, _ in self.module.named_parameters():
            if name in self.fast_params:
                yield self.fast_params[name]

def update_fast_params(fmodule: FunctionalModule, grads, lr):
    """
    Update the fast parameters using SGD, ensuring proper gradient application.
    """
    updated = {}
    for (name, param), grad in zip(fmodule.fast_params.items(), grads):
        if grad is not None:
            updated[name] = param - lr * grad
        else:
            updated[name] = param.clone()
    fmodule.fast_params = updated

@contextmanager
def functional_context(module: nn.Module):
    """
    Enhanced context manager that ensures proper cleanup of functional module.
    """
    fmodule = FunctionalModule(module)
    try:
        yield fmodule
    finally:
        # Clean up by explicitly deleting the functional module
        del fmodule.fast_params
        del fmodule.module
    
class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')
        betas = torch.cat([torch.zeros(1), betas], dim=0)  # Padding beta_0 = 0

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.shape[0]):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.shape[0]):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.num_steps = num_steps
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class InnerLoopOptimizer:
    def __init__(self, initial_lr=1e-2, warmup_steps=1, decay_rate=0.9):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        
    def get_lr(self, step, loss_history):
        # Implement warmup
        if step < self.warmup_steps:
            return self.initial_lr * ((step + 1) / self.warmup_steps)
            
        # Implement decay based on loss improvement
        if len(loss_history) >= 2:
            loss_improvement = (loss_history[-2] - loss_history[-1]) / loss_history[-2]
            if loss_improvement < 0.01:  # If improvement is small
                return self.initial_lr * (self.decay_rate ** (step - self.warmup_steps))
        
        return self.initial_lr


class DitTalkingHead(nn.Module):
    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=76, fps=25, n_motions=100, n_prev_motions=10, 
                 audio_model="hubert", feature_dim=512, n_diff_steps=500, diff_schedule="cosine", 
                 cfg_mode="incremental", guiding_conditions="audio,"):
        super().__init__()

        # Model parameters
        self.target = target # 预测原始图像还是预测噪声
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim # motion 特征维度
        self.fps = fps
        self.n_motions = n_motions # 当前motion100个, window_length, T_w
        self.n_prev_motions = n_prev_motions # 前续motion 10个, T_p
        self.feature_dim = feature_dim

        # Audio encoder
        self.audio_model = audio_model
        if self.audio_model == 'wav2vec2':
            print("using wav2vec2 audio encoder ...")
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert': # 根据经验，hubert特征提取器效果更好
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh': # 根据经验，hubert特征提取器效果更好
            print("using hubert chinese ori")
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, feature_dim)
            self.start_audio_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, feature_dim))
        else:
            raise ValueError(f'Unknown architecture {architecture}!')

        self.start_motion_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.motion_feat_dim)) # 1, 10, 76

        # Diffusion model
        self.denoising_net = DenoisingNetwork(device=device, n_motions=self.n_motions, n_prev_motions=self.n_prev_motions, 
                                              motion_feat_dim=self.motion_feat_dim, feature_dim=feature_dim)
        # diffusion schedule
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        # Classifier-free settings
        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['audio']]
        if 'audio' in self.guiding_conditions:
            audio_feat_dim = feature_dim
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, audio_feat_dim)) # 1, 1, 512

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device
    import math
    
    def inner_loop_adaptation(self, f_denoise, motion_feat_noisy, audio_feat, 
                            prev_motion_feat, prev_audio_feat, time_step, indicator,
                            motion_feat, num_inner_steps=10):  # Increased from 3 to 10
        """Optimized inner-loop adaptation with more steps and adaptive stopping"""
        
        torch.autograd.set_detect_anomaly(True)
        
        # Set up input tensors
        motion_feat_noisy = motion_feat_noisy.detach().requires_grad_(True)
        audio_feat = audio_feat.detach().requires_grad_(True)
        prev_motion_feat = prev_motion_feat.detach().requires_grad_(True)
        prev_audio_feat = prev_audio_feat.detach().requires_grad_(True)
        motion_feat = motion_feat.detach().requires_grad_(True)
        
        # Initialize optimizer parameters with cosine learning rate schedule
        initial_lr = 2e-2  # Slightly increased initial learning rate
        min_lr = 1e-3
        momentum_factor = 0.9
        patience = 3  # Number of steps to wait before early stopping
        min_improvement = 1e-4  # Minimum improvement threshold
        
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        # Create parameter copies with gradient tracking
        param_dict = {}
        momentum_dict = {}
        best_params = {}
        
        for name, param in f_denoise.named_parameters():
            if param.requires_grad:
                param_dict[name] = param.detach().clone().requires_grad_(True)
                momentum_dict[name] = torch.zeros_like(param.data)
                best_params[name] = param_dict[name].clone()
        
        for step in range(num_inner_steps):
            try:
                # Calculate cosine learning rate
                progress = step / num_inner_steps
                current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(progress * math.pi))
                
                # Forward pass with gradient tracking
                with torch.enable_grad():
                    pred = f_denoise(motion_feat_noisy, audio_feat, 
                                prev_motion_feat, prev_audio_feat, 
                                time_step, indicator)
                    
                    pred_current = pred[:, -self.n_motions:, :]
                    
                    # Compute loss with L1 regularization
                    mse_loss = F.mse_loss(pred_current, motion_feat)
                    l1_reg = 1e-5 * sum(p.abs().sum() for name, p in param_dict.items())
                    smoothness = 1e-4 * torch.mean(torch.abs(pred_current[:, 1:] - pred_current[:, :-1]))
                    
                    loss_inner = mse_loss + l1_reg + smoothness
                    
                    current_loss = loss_inner.item()
                    loss_history.append(current_loss)
                    
                    logger.debug(f"Step {step}:")
                    logger.debug(f"  Loss: {current_loss:.6f}")
                    logger.debug(f"  Learning rate: {current_lr:.6f}")
                    
                    # Check for improvement
                    if current_loss < best_loss - min_improvement:
                        best_loss = current_loss
                        patience_counter = 0
                        # Save best parameters
                        for name, param in param_dict.items():
                            best_params[name] = param.clone()
                    else:
                        patience_counter += 1
                    
                    # Early stopping check
                    if patience_counter >= patience:
                        logger.debug(f"Early stopping at step {step}")
                        break
                    
                    # Compute and apply gradients
                    trainable_params = []
                    param_names = []
                    for name, param in param_dict.items():
                        trainable_params.append(param)
                        param_names.append(name)
                    
                    grads = torch.autograd.grad(
                        loss_inner,
                        trainable_params,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )
                    
                    # Update parameters
                    with torch.no_grad():
                        for name, param, grad in zip(param_names, trainable_params, grads):
                            if grad is not None:
                                # Momentum update
                                new_momentum = momentum_factor * momentum_dict[name] + (1 - momentum_factor) * grad
                                momentum_dict[name] = new_momentum
                                
                                # Parameter update
                                new_param = param - current_lr * new_momentum
                                param_dict[name] = new_param.clone().requires_grad_(True)
                                
                                # Update model parameter
                                orig_param = dict(f_denoise.named_parameters())[name]
                                orig_param.data.copy_(new_param.data)
            
            except Exception as e:
                logger.error(f"Error during step {step}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        # Restore best parameters
        with torch.no_grad():
            for name, param in f_denoise.named_parameters():
                if name in best_params:
                    param.data.copy_(best_params[name].data)
        
        # Final forward pass
        with torch.enable_grad():
            motion_feat_target = f_denoise(motion_feat_noisy, audio_feat,
                                        prev_motion_feat, prev_audio_feat,
                                        time_step, indicator)
        
        torch.autograd.set_detect_anomaly(False)
        return motion_feat_target


    # def inner_loop_adaptation(self, f_denoise, motion_feat_noisy, audio_feat, 
    #                         prev_motion_feat, prev_audio_feat, time_step, indicator,
    #                         motion_feat, num_inner_steps=3):
    #     """Optimized inner-loop adaptation with better parameter tracking"""
        
    #     torch.autograd.set_detect_anomaly(True)
        
    #     def debug_tensor(x, name):
    #         if torch.is_tensor(x):
    #             logger.debug(f"{name}:")
    #             logger.debug(f"  shape: {x.shape}")
    #             logger.debug(f"  requires_grad: {x.requires_grad}")
    #             logger.debug(f"  grad_fn: {x.grad_fn}")
    #             logger.debug(f"  is_leaf: {x.is_leaf}")
        
    #     # Set up input tensors
    #     logger.debug("Setting up input tensors...")
    #     motion_feat_noisy = motion_feat_noisy.detach().requires_grad_(True)
    #     audio_feat = audio_feat.detach().requires_grad_(True)
    #     prev_motion_feat = prev_motion_feat.detach().requires_grad_(True)
    #     prev_audio_feat = prev_audio_feat.detach().requires_grad_(True)
    #     motion_feat = motion_feat.detach().requires_grad_(True)
        
    #     debug_tensor(motion_feat_noisy, "motion_feat_noisy")
    #     debug_tensor(audio_feat, "audio_feat")
    #     debug_tensor(prev_motion_feat, "prev_motion_feat")
    #     debug_tensor(prev_audio_feat, "prev_audio_feat")
    #     debug_tensor(motion_feat, "motion_feat")
        
    #     # Initialize optimizer parameters
    #     initial_lr = 1e-2
    #     warmup_steps = 1
    #     decay_rate = 0.9
    #     momentum_factor = 0.9
        
    #     loss_history = []
        
    #     # Create parameter copies with gradient tracking
    #     logger.debug("Creating parameter copies...")
    #     param_dict = {}
    #     momentum_dict = {}
    #     for name, param in f_denoise.named_parameters():
    #         if param.requires_grad:
    #             # Create new parameter with gradient tracking
    #             param_dict[name] = param.detach().clone().requires_grad_(True)
    #             momentum_dict[name] = torch.zeros_like(param.data)
    #             debug_tensor(param_dict[name], f"param_dict[{name}]")
        
    #     # Assign initial parameter values
    #     with torch.no_grad():
    #         for name, param in f_denoise.named_parameters():
    #             if param.requires_grad:
    #                 param.data.copy_(param_dict[name].data)
        
    #     for step in range(num_inner_steps):
    #         logger.debug(f"\nStarting inner loop step {step}")
    #         try:
    #             # Enable gradient tracking for forward pass
    #             with torch.enable_grad():
    #                 # Forward pass
    #                 pred = f_denoise(motion_feat_noisy, audio_feat, 
    #                             prev_motion_feat, prev_audio_feat, 
    #                             time_step, indicator)
                    
    #                 debug_tensor(pred, "pred")
                    
    #                 # Extract current predictions
    #                 pred_current = pred[:, -self.n_motions:, :]
                    
    #                 debug_tensor(pred_current, "pred_current")
                    
    #                 # Compute loss
    #                 loss_inner = F.mse_loss(pred_current, motion_feat)
                    
    #                 # Add L2 regularization
    #                 l2_reg = sum(p.pow(2).sum() for name, p in param_dict.items())
    #                 loss_inner = loss_inner + 1e-5 * l2_reg
                    
    #                 debug_tensor(loss_inner, "loss_inner")
                    
    #                 logger.debug(f"Step {step}:")
    #                 logger.debug(f"  Loss value: {loss_inner.item():.6f}")
                    
    #                 # Store loss history
    #                 loss_history.append(loss_inner.item())
                    
    #                 # Calculate learning rate
    #                 current_lr = initial_lr
    #                 if step >= warmup_steps and len(loss_history) >= 2:
    #                     loss_improvement = (loss_history[-2] - loss_history[-1]) / (loss_history[-2] + 1e-8)
    #                     if loss_improvement < 0.01:
    #                         current_lr *= decay_rate
                    
    #                 logger.debug("Computing gradients...")
    #                 # Get parameters that require gradients
    #                 trainable_params = []
    #                 param_names = []
    #                 for name, param in f_denoise.named_parameters():
    #                     if param.requires_grad:
    #                         trainable_params.append(param_dict[name])
    #                         param_names.append(name)
    #                         debug_tensor(param_dict[name], f"trainable_param[{name}]")
                    
    #                 # Compute gradients
    #                 grads = torch.autograd.grad(
    #                     loss_inner,
    #                     trainable_params,
    #                     create_graph=True,
    #                     retain_graph=True,
    #                     allow_unused=True
    #                 )
                    
    #                 logger.debug("Updating parameters...")
    #                 # Update parameters
    #                 with torch.no_grad():
    #                     for name, param, grad in zip(param_names, trainable_params, grads):
    #                         if grad is not None:
    #                             # Update momentum
    #                             new_momentum = momentum_factor * momentum_dict[name] + (1 - momentum_factor) * grad
    #                             momentum_dict[name] = new_momentum
                                
    #                             # Update parameter
    #                             new_param = param - current_lr * new_momentum
    #                             param_dict[name] = new_param.clone().requires_grad_(True)
                                
    #                             # Update original parameter
    #                             orig_param = dict(f_denoise.named_parameters())[name]
    #                             orig_param.data.copy_(new_param.data)
                                
    #                             debug_tensor(param_dict[name], f"updated_param[{name}]")
            
    #         except Exception as e:
    #             logger.error(f"Error during step {step}: {str(e)}")
    #             logger.error(f"Traceback: {traceback.format_exc()}")
    #             raise
        
    #     logger.debug("Computing final output...")
    #     # Final forward pass
    #     with torch.enable_grad():
    #         motion_feat_target = f_denoise(motion_feat_noisy, audio_feat,
    #                                     prev_motion_feat, prev_audio_feat,
    #                                     time_step, indicator)
    #         debug_tensor(motion_feat_target, "motion_feat_target")
        
    #     torch.autograd.set_detect_anomaly(False)
    #     return motion_feat_target

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_coef) motion coefficients or features
            audio_or_feat: (N, L_audio) raw audio or audio feature
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion coefficients or feature
            prev_audio_feat: (N, n_prev_motions, d_audio) previous audio features
            time_step: (N,)
            indicator: (N, L) 0/1 indicator of real (unpadded) motion coefficients

        Returns:
           motion_feat_noise: (N, L, d_motion)
        """
        batch_size = motion_feat.shape[0]

        # 加载语音特征
        if audio_or_feat.ndim == 2: # 原始语音
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3: # 语音特征
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        # 前续motion特征
        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        
        # 前续语音特征
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        # Classifier-free guidance
        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = torch.rand(batch_size, device=self.device) < null_cond_prob
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)
            else:
                # len(self.guiding_conditions) > 1 and self.cfg_mode == 'incremental'
                # full (0.45), w/o style (0.45), w/o style or audio (0.1)
                mask_flag = torch.rand(batch_size, device=self.device)
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag > 0.9
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)

        if time_step is None:
            # Sample time step
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # (N,)

        # The forward diffusion process
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # (N,)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (N, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (N, 1, 1)

        eps = torch.randn_like(motion_feat)  # (N, L, d_motion)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        # The reverse diffusion process
        # motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, 
        #                                         prev_motion_feat, prev_audio_feat, time_step, indicator)

        # --- Inner-loop adaptation on the denoising network ---
        # We want to update the denoising network’s parameters via a differentiable inner loop.
        # Since the denoising network (DenoisingNetwork) expects an input that is a concatenation of
        # prev_motion_feat and current motion_feat, its output has shape (batch_size, n_prev_motions+n_motions, d_motion).
        # Our target for inner-loop loss is the current motion features (motion_feat of shape (batch_size, n_motions, d_motion)).
        num_inner_steps = 3   # number of inner updates
        inner_lr = 1e-2       # inner-loop learning rate

       # --- Inside DitTalkingHead.forward, in the inner-loop update block ---

        # with functional_context(self.denoising_net) as f_denoise:
        #     logger.info("[bold green]Starting inner-loop update for denoising_net functional copy.[/]")
            
        #     for step in range(num_inner_steps):
        #         # Forward pass with current fast parameters
        #         pred = f_denoise(motion_feat_noisy, audio_feat, 
        #                         prev_motion_feat, prev_audio_feat, 
        #                         time_step, indicator)
                
        #         # Extract current motion predictions
        #         pred_current = pred[:, -self.n_motions:, :]
                
        #         # Compute loss
        #         loss_inner = F.mse_loss(pred_current, motion_feat)
        #         logger.debug(f"Inner loop step {step}: loss = {loss_inner.item():.6f}")
                
        #         # Get list of parameters that require gradients
        #         trainable_params = [p for p in f_denoise.parameters() if p.requires_grad]
                
        #         try:
        #             # First try with allow_unused=False to catch any issues
        #             grads = torch.autograd.grad(
        #                 loss_inner,
        #                 trainable_params,
        #                 create_graph=True,
        #                 allow_unused=True  # Changed to True to handle unused parameters
        #             )
                    
        #             # Replace None gradients with zeros
        #             grads = [torch.zeros_like(p) if g is None else g 
        #                     for p, g in zip(trainable_params, grads)]
                    
        #         except Exception as e:
        #             logger.error(f"Error during gradient computation: {str(e)}")
        #             # If gradient computation fails, return zero gradients
        #             grads = [torch.zeros_like(p) for p in trainable_params]
                
        #         # Update fast parameters
        #         update_fast_params(f_denoise, grads, inner_lr)
            
        #     # Final forward pass with adapted parameters
        #     motion_feat_target = f_denoise(motion_feat_noisy, audio_feat,
        #                                 prev_motion_feat, prev_audio_feat,
        #                                 time_step, indicator)
# In DitTalkingHead's forward method:
        # In DitTalkingHead's forward method:
        # In DitTalkingHead's forward method:
        with functional_context(self.denoising_net) as f_denoise:
            logger.info("[bold green]Starting inner-loop update for denoising_net functional copy.[/]")
            try:
                # Ensure all inputs are on the correct device
                # motion_feat_noisy = to_device(motion_feat_noisy, "motion_feat_noisy", self.device)
                # audio_feat = to_device(audio_feat, "audio_feat", self.device)
                # prev_motion_feat = to_device(prev_motion_feat, "prev_motion_feat", self.device)
                # prev_audio_feat = to_device(prev_audio_feat, "prev_audio_feat", self.device)
                # time_step = to_device(time_step, "time_step", self.device)
                # motion_feat = to_device(motion_feat, "motion_feat", self.device)
                
                # Get target from inner loop adaptation
                motion_feat_target = self.inner_loop_adaptation(
                    f_denoise=f_denoise,
                    motion_feat_noisy=motion_feat_noisy,
                    audio_feat=audio_feat,
                    prev_motion_feat=prev_motion_feat,
                    prev_audio_feat=prev_audio_feat,
                    time_step=time_step,
                    indicator=indicator,
                    motion_feat=motion_feat
                )
                
                # Return all required values for training
                return motion_feat_noisy, motion_feat_target, prev_motion_feat, prev_audio_feat
                
            except Exception as e:
                logger.error(f"Error in forward pass: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

        def to_device(x, name, device):
            """Helper function to safely convert inputs to tensors on the correct device"""
            logger.debug(f"Converting {name} of type {type(x)} to tensor on device {device}")
            if torch.is_tensor(x):
                return x.to(device)
            elif isinstance(x, list):
                return torch.tensor(x, device=device)
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            else:
                logger.error(f"Unexpected type for {name}: {type(x)}")
                raise TypeError(f"Cannot convert {name} of type {type(x)} to tensor")
            
    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions

        # # Strategy 1: resample during audio feature extraction
        # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

        # Strategy 2: resample after audio feature extraction (BackResample)
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)

        audio_feat = self.audio_feature_map(hidden_states)  # (N, L, feature_dim)
        return audio_feat

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False):
        # Check and convert inputs
        batch_size = audio_or_feat.shape[0]

        # Check CFG conditions
        if cfg_mode is None:  # Use default CFG mode
            cfg_mode = self.cfg_mode
        if cfg_cond is None:  # Use default CFG conditions
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', ]]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # sort cfg_cond and cfg_scale
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', ].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        # Prepare input for the reverse diffusion process (including optional classifier-free guidance)
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        audio_feat_in = [audio_feat_null]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(motion_at_T)
            else:
                z = torch.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in)

            # Apply thresholding if specified
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)

            # Unconditional target (CFG) or the conditional target (non-CFG)
            target_theta = results[0][:, -self.n_motions:]
            # Classifier-free Guidance (optional)
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        else:
            return traj[0], motion_at_T, audio_feat


class DenoisingNetwork(nn.Module):
    def __init__(self, device='cuda', motion_feat_dim=76, 
                 use_indicator=None, architecture="decoder", feature_dim=512, n_heads=8, 
                 n_layers=8, mlp_ratio=4, align_mask_width=1, no_use_learnable_pe=True, n_prev_motions=10,
                 n_motions=100, n_diff_steps=500, ):
        super().__init__()

        # Model parameters
        self.motion_feat_dim = motion_feat_dim 
        self.use_indicator = use_indicator

        # Transformer
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe

        # sequence length
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        # Temporal embedding for the diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        if self.use_learnable_pe:
            # Learnable positional encoding
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        # Transformer decoder
        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),
                                          self.feature_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
                activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len, frame_width=1, expansion=self.align_mask_width - 1)
                # print(f"alignment_mask: ", alignment_mask.shape)
                # alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
                self.register_buffer('alignment_mask', alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Motion decoder
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),
            # nn.Tanh() # 增加了一个tanh
            # nn.Softmax()
        )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_motion). Noisy motion feature
            audio_feat: (N, L, feature_dim)
            prev_motion_feat: (N, L_p, d_motion). Padded previous motion coefficients or feature
            prev_audio_feat: (N, L_p, d_audio). Padded previous motion coefficients or feature
            step: (N,)
            indicator: (N, L). 0/1 indicator for the real (unpadded) motion feature

        Returns:
            motion_feat_target: (N, L_p + L, d_motion)
        """
        # Diffusion time step embedding
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)  # (N, 1, diff_step_dim)

        if indicator is not None:
            indicator = torch.cat([torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                                   indicator], dim=1)  # (N, L_p + L)
            indicator = indicator.unsqueeze(-1)  # (N, L_p + L, 1)

        # Concat features and embeddings
        if self.architecture == 'decoder':
            # print("prev_motion_feat: ", prev_motion_feat.shape, "motion_feat: ", motion_feat.shape)
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)  # (N, L_p + L, d_motion)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)  # (N, L_p + L, d_motion + d_audio + 1)

        feats_in = self.feature_proj(feats_in)  # (N, L_p + L, feature_dim)
        # feats_in = torch.cat([person_feat, feats_in], dim=1)  # (N, 1 + L_p + L, feature_dim)

        if self.use_learnable_pe:
            # feats_in = feats_in + self.PE
            feats_in = feats_in + self.PE + diff_step_embedding
        else:
            # feats_in = self.PE(feats_in)
            feats_in = self.PE(feats_in) + diff_step_embedding

        # Transformer
        if self.architecture == 'decoder':
            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (N, L_p + L, d_audio)
            # print(f"feats_in: {feats_in.shape}, audio_feat_in: {audio_feat_in.shape}, memory_mask: {self.alignment_mask.shape}")
            feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Decode predicted motion feature noise / sample
        # motion_feat_target = self.motion_dec(feat_out[:, 1:])  # (N, L_p + L, d_motion)
        motion_feat_target = self.motion_dec(feat_out)  # (N, L_p + L, d_motion)

        return motion_feat_target

if __name__ == "__main__":
    device = "cuda"
    motion_feat_dim = 76
    n_motions = 100 # L
    n_prev_motions = 10 # L_p

    L_audio = int(16000 * n_motions / 25) # 64000
    d_audio = 768

    N = 5
    feature_dim = 512

    motion_feat = torch.ones((N, n_motions, motion_feat_dim)).to(device)
    prev_motion_feat = torch.ones((N, n_prev_motions, motion_feat_dim)).to(device)

    audio_or_feat = torch.ones((N, L_audio)).to(device)
    prev_audio_feat = torch.ones((N, n_prev_motions, d_audio)).to(device)

    time_step = torch.ones(N, dtype=torch.long).to(device)

    model = DitTalkingHead().to(device)

    z = model(motion_feat, audio_or_feat, prev_motion_feat=None, 
              prev_audio_feat=None, time_step=None, indicator=None)
    traj, motion_at_T, audio_feat = z[0], z[1], z[2]
    print(motion_at_T.shape, audio_feat.shape)