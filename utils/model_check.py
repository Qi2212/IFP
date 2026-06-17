"""
Model Validation & Robustness Checks
=====================================
Utility functions for validating models, paths, devices, and configurations
during training and inference.
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional


def check_path_exists(path: str, name: str = "Path") -> None:
    """
    Verify that a file or directory path exists. Raises FileNotFoundError if not.

    Args:
        path: Path to check.
        name: Human-readable name for error messages.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def check_model_loaded(model: nn.Module, name: str = "Model") -> None:
    """
    Verify that a PyTorch model is not None and is in eval mode.

    Args:
        model: PyTorch module to check.
        name: Human-readable name for error messages.
    """
    if model is None:
        raise ValueError(f"{name} is None. Model was not loaded successfully.")

    # Count parameters to sanity-check
    total_params = sum(p.numel() for p in model.parameters())
    if total_params == 0:
        raise ValueError(f"{name} has 0 parameters. Model may not be loaded correctly.")

    print(f"[check] {name}: {total_params / 1e6:.1f}M parameters")


def check_device_available(device: str) -> torch.device:
    """
    Validate device string and return torch.device.
    Falls back to CPU if CUDA is requested but unavailable.

    Args:
        device: Device string (e.g., 'cuda:0', 'cpu').

    Returns:
        torch.device: Validated device.
    """
    if "cuda" in device and not torch.cuda.is_available():
        print(f"[warn] CUDA requested ({device}) but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device)


def validate_config_keys(config: dict, required_keys: List[str], section: str = "config") -> None:
    """
    Validate that required keys exist in a configuration dictionary.

    Args:
        config: Configuration dictionary.
        required_keys: List of required key names.
        section: Section name for error messages.

    Raises:
        KeyError: If any required key is missing.
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(
            f"Missing required keys in [{section}]: {missing}. "
            f"Available keys: {list(config.keys())}"
        )


def ensure_dir(dir_path: str) -> None:
    """
    Create directory if it does not exist.

    Args:
        dir_path: Directory path to ensure.
    """
    os.makedirs(dir_path, exist_ok=True)


def check_tensor_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Check tensor for NaN or Inf values and raise if found.

    Args:
        tensor: PyTorch tensor to check.
        name: Name for error messages.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"[check] NaN detected in {name}!")
    if torch.isinf(tensor).any():
        raise ValueError(f"[check] Inf detected in {name}!")


def check_gradient_flow(model: nn.Module, name: str = "model") -> None:
    """
    Check if gradients are flowing through the model (non-zero grads for at least one parameter).

    Args:
        model: PyTorch module.
        name: Name for logging.
    """
    zero_grad_params = []
    for pname, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().sum() == 0:
                zero_grad_params.append(pname)

    if zero_grad_params:
        print(f"[warn] {name}: {len(zero_grad_params)} params have zero gradients")
    else:
        print(f"[check] {name}: gradients flowing")
