"""
DINO Model Loader (Unified DINOv2 / DINOv3)
============================================
Provides a unified interface for loading DINOv2 and DINOv3 models
with local pretrained weights.

Supports:
  DINOv2: vit_small (vits14), vit_base (vitb14), vit_large (vitl14), vit_giant2 (vitg14)
  DINOv3: dinov3_vith16plus (dino3h), dinov3_vitl16 (dino3l)
"""

import os
import sys
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Patch size helpers
# ---------------------------------------------------------------------------

def get_patch_size(model: nn.Module) -> int:
    """
    Infer patch size from a DINO model.

    Args:
        model: DINOv2 or DINOv3 model instance.

    Returns:
        int: Patch size (e.g., 14 for DINOv2, 16 for DINOv3).
    """
    if hasattr(model, "patch_size"):
        ps = model.patch_size
        return int(ps[0] if isinstance(ps, (tuple, list)) else ps)

    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        ps = model.patch_embed.patch_size
        return int(ps[0] if isinstance(ps, (tuple, list)) else ps)

    return 16  # default for DINOv3


# ---------------------------------------------------------------------------
# DINOv2 loading
# ---------------------------------------------------------------------------

DINOV2_WEIGHTS_MAP = {
    "dino2b": "dinov2_vitb14_pretrain.pth",
    "dino2l": "dinov2_vitl14_pretrain.pth",
    "dino2g": "dinov2_vitg14_pretrain.pth",
}


def _load_dinov2_model(dino_variant: str, dino_root: str, weights_dir: str) -> nn.Module:
    """
    Load a DINOv2 model using hubconf from the local dinov2 directory.

    Args:
        dino_variant: One of 'dino2b', 'dino2l', 'dino2g'.
        dino_root: Path to the dinov2 repository root.
        weights_dir: Directory containing pretrained .pth files.

    Returns:
        nn.Module: Loaded DINOv2 model.
    """
    # Add dinov2 root to path for hubconf import
    if dino_root not in sys.path:
        sys.path.insert(0, dino_root)

    import hubconf

    if dino_variant == "dino2g":
        model = hubconf.dinov2_vitg14(pretrained=False)
    elif dino_variant == "dino2l":
        model = hubconf.dinov2_vitl14(pretrained=False)
    elif dino_variant == "dino2b":
        model = hubconf.dinov2_vitb14(pretrained=False)
    else:
        raise ValueError(f"Unsupported DINOv2 variant: {dino_variant}. Choose from: dino2b, dino2l, dino2g")

    weights_file = DINOV2_WEIGHTS_MAP[dino_variant]
    weights_path = os.path.join(weights_dir, weights_file)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"DINOv2 weights not found: {weights_path}. "
            f"Please download and place in {weights_dir}"
        )

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    return model


# ---------------------------------------------------------------------------
# DINOv3 loading
# ---------------------------------------------------------------------------

DINOV3_MODEL_NAME_MAP = {
    "dino3h": "dinov3_vith16plus",
    "dino3l": "dinov3_vitl16",
}

DINOV3_WEIGHTS_MAP = {
    "dino3h": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dino3l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _load_dinov3_model(dino_variant: str, dino_root: str, weights_dir: str) -> nn.Module:
    """
    Load a DINOv3 model from local repository using torch.hub.

    Args:
        dino_variant: One of 'dino3h', 'dino3l'. Default is 'dino3h'.
        dino_root: Path to the dinov3 repository root.
        weights_dir: Directory containing pretrained .pth files.

    Returns:
        nn.Module: Loaded DINOv3 model.
    """
    model_name = DINOV3_MODEL_NAME_MAP.get(dino_variant)
    if model_name is None:
        raise ValueError(f"Unsupported DINOv3 variant: {dino_variant}. Choose from: dino3h, dino3l")

    weights_file = DINOV3_WEIGHTS_MAP[dino_variant]
    weights_path = os.path.join(weights_dir, weights_file)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"DINOv3 weights not found: {weights_path}. "
            f"Please download and place in {weights_dir}"
        )

    model = torch.hub.load(
        repo_or_dir=dino_root,
        model=model_name,
        source="local",
        pretrained=False,
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    return model


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def load_dino_model(
    dino_variant: str = "dino3h",
    dino_root: str = "./dinov3",
    weights_dir: str = "./dinov3/download_models",
    device: str = "cuda:0",
) -> nn.Module:
    """
    Unified interface to load a DINOv2 or DINOv3 model.

    Args:
        dino_variant: Model variant identifier.
            DINOv3: 'dino3h' (ViT-H/16+, default), 'dino3l' (ViT-L/16)
            DINOv2: 'dino2b' (ViT-B/14), 'dino2l' (ViT-L/14), 'dino2g' (ViT-G/14)
        dino_root: Root directory of dino repository.
        weights_dir: Directory containing pretrained weight files.
        device: Target device.

    Returns:
        nn.Module: Loaded DINO model in eval mode on the target device.
    """
    if dino_variant.startswith("dino3"):
        model = _load_dinov3_model(dino_variant, dino_root, weights_dir)
    elif dino_variant.startswith("dino2"):
        model = _load_dinov2_model(dino_variant, dino_root, weights_dir)
    else:
        raise ValueError(
            f"Unknown DINO variant: {dino_variant}. "
            f"Must start with 'dino2' or 'dino3'."
        )

    model.to(device)
    model.eval()
    return model


def extract_tokens_from_dino(
    model: nn.Module,
    pil_img,
    dino_variant: str = "dino3h",
    resize: tuple = (512, 512),
    device: str = "cuda:0",
) -> tuple:
    """
    Extract patch tokens from a PIL image using a DINO model.

    Supports both DINOv2 (forward_features) and DINOv3 (get_intermediate_layers).

    Args:
        model: Loaded DINO model (on target device).
        pil_img: PIL Image in RGB mode.
        dino_variant: 'dino2*' or 'dino3*' to select extraction method.
        resize: (H, W) tuple for image resize.
        device: Device string.

    Returns:
        tuple: (tokens_np [N, D], (h, w)) where h, w are the spatial grid dimensions.
    """
    from torchvision import transforms

    img = pil_img.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    inp = preprocess(img).unsqueeze(0).to(device)

    patch_size = get_patch_size(model)
    h = resize[0] // patch_size
    w = resize[1] // patch_size
    expected_n = h * w

    with torch.no_grad():
        if dino_variant.startswith("dino3"):
            # DINOv3: use get_intermediate_layers
            out = model.get_intermediate_layers(inp, n=1)[0]
            if out.shape[1] == expected_n + 1:
                tokens = out[:, 1:, :]
            else:
                tokens = out
            tokens = tokens.squeeze(0).cpu().numpy()

        elif dino_variant.startswith("dino2"):
            # DINOv2: use forward_features
            feats = model.forward_features(inp)
            if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
                tokens = feats["x_norm_patchtokens"].squeeze(0).cpu().numpy()
            else:
                raise ValueError("DINOv2 forward_features missing x_norm_patchtokens")
        else:
            raise ValueError(f"Unknown dino_variant: {dino_variant}")

    # Handle token count mismatch
    if tokens.shape[0] == expected_n + 1:
        tokens = tokens[1:]

    if tokens.shape[0] > expected_n:
        tokens = tokens[:expected_n]
    elif tokens.shape[0] < expected_n:
        need = expected_n - tokens.shape[0]
        pad = np_repeat_last(tokens, need)
        tokens = np.concatenate([tokens, pad], axis=0)

    return tokens, (h, w)


def np_repeat_last(arr, n: int):
    """Repeat the last row of a numpy array n times."""
    import numpy as np
    return np.repeat(arr[-1:], n, axis=0)
