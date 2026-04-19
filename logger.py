import logging
from rich.logging import RichHandler
import warnings

import os

# "The quieter you become, the more you are able to hear"


import torch
import torch.autograd.profiler as profiler
import torch.autograd.anomaly_mode as anomaly_mode
import logging
import sys
from typing import Dict, Any, Optional

import logging
from rich.logging import RichHandler
import torch
import torch.autograd.profiler as profiler
import sys
from typing import Dict, Any, Optional
from rich.traceback import install



    
# log_level = logging.WARNING    
log_level = logging.INFO
# log_level = logging.DEBUG

try:
    # Silence third-party loggers
    for module in [
        'paramiko', 'albumentations', 'torch.cuda.amp',
        'transformers', 'tensorflow', 'absl', 'numexpr',
        'matplotlib', 'PIL', 'h5py', 'oauth2client',
        'torch.utils.data.dataloader'
    ]:
        logging.getLogger(module).setLevel(logging.ERROR)

except Exception:
    pass




# Silence TensorFlow
# Try to silence TensorFlow only if it's installed
try:
    import tensorflow as tf
    import os
    
    # Silence TensorFlow
    tf.get_logger().setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

except ImportError:
    pass


# Configure warning filters
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    # Configure absl logging to suppress logs
    import absl.logging
    # Suppress Abseil logs
    absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
    absl.logging.set_verbosity(absl.logging.FATAL)
    absl.logging.set_stderrthreshold(absl.logging.FATAL)
except Exception:
    pass


# Specific warning suppressions
warnings.filterwarnings(
    'ignore', 
    message='torch.cuda.amp.autocast.*',
    category=FutureWarning
)
warnings.filterwarnings(
    'ignore',
    message='.*gradient_checkpointing.*',
    category=UserWarning
)
warnings.filterwarnings(
    'ignore',
    message='.*weights_only.*',
    category=UserWarning
)




# Configure logging with both RichHandler and FileHandler
file_handler = logging.FileHandler("project.log")  # Log to a file
file_handler.setLevel(log_level)

 # Configure rich handler
rich_handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_time=True,
    show_path=True
)
rich_handler.setLevel(log_level)

logging.basicConfig(
    level=log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich_handler, file_handler]
)

# Shared logger
logger = logging.getLogger("vasa")
logger.setLevel(log_level)


from rich.traceback import install
# Enable rich tracebacks globally
install()



class TorchDebugger:
    """Helper class to enable comprehensive PyTorch debugging with version compatibility"""
    
    def __init__(
        self,
        log_level: int = logging.DEBUG,
        enable_anomaly_detection: bool = True,
        enable_grad_debugging: bool = True
    ):
            
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_grad_debugging = enable_grad_debugging
        
        if enable_grad_debugging:
            torch._C._debug_set_autodiff_subgraph_inlining(False)

    def debug_tensor(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        detailed: bool = True
    ) -> None:
        """Print detailed tensor information with version compatibility"""
        try:
            # Basic info available in all versions
            basic_info = {
                "name": name,
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "device": tensor.device,
                "requires_grad": tensor.requires_grad,
                "has_grad_fn": hasattr(tensor, 'grad_fn'),
                "grad_fn": tensor.grad_fn if hasattr(tensor, 'grad_fn') else None,
                "is_leaf": tensor.is_leaf,
                "grad": tensor.grad
            }
            
            # Version-independent detailed info
            if detailed:
                detailed_info = {
                    "numel": tensor.numel(),
                    "stride": tensor.stride(),
                    "is_contiguous": tensor.is_contiguous(),
                }
                
                # Try to get newer attributes safely
                try:
                    detailed_info["layout"] = tensor.layout
                except AttributeError:
                    pass
                
                # Add statistics for floating point tensors
                if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                    with torch.no_grad():
                        detailed_info.update({
                            "min": tensor.min().item(),
                            "max": tensor.max().item(),
                            "mean": tensor.mean().item(),
                            "std": tensor.std().item(),
                            "has_nan": torch.isnan(tensor).any().item(),
                            "has_inf": torch.isinf(tensor).any().item(),
                            "abs_mean": torch.abs(tensor).mean().item(),
                        })
                
                basic_info.update(detailed_info)
            
            # Print info
            logger.debug(f"\n[bold blue]Tensor Debug Info for {name}:[/]")
            for k, v in basic_info.items():
                if k in ["grad_fn", "grad"] and v is not None:
                    logger.debug(f"  [yellow]{k}[/]: {type(v).__name__}")
                else:
                    logger.debug(f"  [yellow]{k}[/]: {v}")
                    
            # Additional gradient information
            if tensor.requires_grad and tensor.grad is not None:
                logger.debug("\n[bold blue]Gradient Info:[/]")
                logger.debug(f"  [yellow]grad shape[/]: {tensor.grad.shape}")
                with torch.no_grad():
                    logger.debug(f"  [yellow]grad mean[/]: {tensor.grad.mean().item():.6f}")
                    logger.debug(f"  [yellow]grad std[/]: {tensor.grad.std().item():.6f}")

        except Exception as e:
            logger.error(f"Error in debug_tensor: {str(e)}")
            
    def debug_model_gradients(self, model: torch.nn.Module) -> None:
        """Debug model gradients"""
        logger.debug("\n[bold blue]Model Gradient Analysis:[/]")
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    with torch.no_grad():
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        grad_norm = param.grad.norm().item()
                        
                    logger.debug(f"\n[yellow]{name}[/]:")
                    logger.debug(f"  Mean: {grad_mean:.6f}")
                    logger.debug(f"  Std: {grad_std:.6f}")
                    logger.debug(f"  Norm: {grad_norm:.6f}")
                else:
                    logger.warning(f"\n[yellow]{name}[/]: No gradient computed!")

    def debug_backward_graph(self, loss: torch.Tensor) -> None:
        """Debug autograd graph from loss"""
        def _print_backward_graph(grad_fn, prefix=""):
            """Recursively print gradient function graph"""
            logger.info(f"{prefix}[cyan]{type(grad_fn).__name__}[/]")
            
            if hasattr(grad_fn, 'next_functions'):
                for next_fn in grad_fn.next_functions:
                    if next_fn[0] is not None:
                        _print_backward_graph(next_fn[0], prefix + "  ")
        
        if hasattr(loss, 'grad_fn'):
            logger.info("\n[bold blue]Backward Graph:[/]")
            _print_backward_graph(loss.grad_fn)
        else:
            logger.warning("[red]Loss tensor has no grad_fn![/]")
            
    def debug_model(
        self,
        model: torch.nn.Module,
        input_shape: Optional[tuple] = None,
        sample_input: Optional[torch.Tensor] = None
    ) -> None:
        """Debug model architecture and parameters"""
        logger.debug("\nModel Debug Info:")
        
        # Print model architecture
        logger.debug("\nModel Architecture:")
        logger.debug(str(model))
        
        # Parameter analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.debug(f"\nParameter Count:")
        logger.debug(f"  Total parameters: {total_params:,}")
        logger.debug(f"  Trainable parameters: {trainable_params:,}")
        
        # Check each parameter
        logger.debug("\nParameter Details:")
        for name, param in model.named_parameters():
            logger.debug(
                f"  {name}: shape={param.shape}, "
                f"requires_grad={param.requires_grad}, "
                f"has_grad={param.grad is not None}"
            )
            
        # If sample input provided, try forward pass
        if sample_input is not None or input_shape is not None:
            try:
                test_input = sample_input if sample_input is not None else \
                            torch.randn(input_shape, device=next(model.parameters()).device)
                with torch.no_grad():
                    output = model(test_input)
                logger.debug(f"\nTest forward pass successful!")
                logger.debug(f"Input shape: {test_input.shape}")
                logger.debug(f"Output shape: {output.shape}")
            except Exception as e:
                logger.error(f"Error in test forward pass: {str(e)}")

    def __enter__(self):
        if self.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(False)

    @staticmethod
    def attach_gradient_hooks(model: torch.nn.Module):
        """Attach gradient hooks to all parameters"""
        hooks = []
        
        def grad_hook(grad):
            with torch.no_grad():
                print(f"Gradient stats - min: {grad.min():.6f}, max: {grad.max():.6f}, mean: {grad.mean():.6f}")
            return grad
            
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(grad_hook)
                hooks.append(hook)
                
        return hooks
    

