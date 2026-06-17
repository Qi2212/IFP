"""
CLIP Model Loader
==================
Loads CLIP models from local paths and provides text feature extraction.
"""

import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple


# CLIP path mapping (relative to project root)
CLIP_PATH_MAP = {
    "vitb": "./clip/clip-vit-base-patch16",
    "vitl": "./clip/clip-vit-large-patch14",
}


def load_clip_model(
    clip_path: str = "./clip/clip-vit-base-patch16",
    device: str = "cuda:0",
) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model and processor from a local path.

    Args:
        clip_path: Path to the CLIP model directory (local or HuggingFace ID).
        device: Target device.

    Returns:
        tuple: (clip_model, clip_processor)
    """
    if not os.path.exists(clip_path):
        raise FileNotFoundError(
            f"CLIP model path not found: {clip_path}. "
            f"Please download CLIP weights. See scripts/download_clip_models.sh"
        )

    clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_path)

    return clip_model, clip_processor


def compute_text_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    texts: List[str],
    device: str = "cuda:0",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute normalized CLIP text features for a list of text strings.

    Args:
        clip_model: Loaded CLIP model.
        clip_processor: CLIP processor.
        texts: List of text strings.
        device: Device string.
        normalize: Whether to L2-normalize the output features.

    Returns:
        Tensor: Text features [B, text_dim], normalized if normalize=True.
    """
    inputs = clip_processor(
        text=texts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        t_feats = clip_model.get_text_features(**inputs)
        if not isinstance(t_feats, torch.Tensor):
            t_feats = t_feats.pooler_output

    if normalize:
        t_feats = F.normalize(t_feats, dim=-1)

    return t_feats


def get_clip_path_by_variant(clip_variant: str) -> str:
    """
    Get the local CLIP model path for a given variant.

    Args:
        clip_variant: 'vitb' (ViT-B/16) or 'vitl' (ViT-L/14).

    Returns:
        str: Local path to the CLIP model.
    """
    if clip_variant not in CLIP_PATH_MAP:
        raise ValueError(
            f"Unsupported CLIP variant: {clip_variant}. "
            f"Choose from: {list(CLIP_PATH_MAP.keys())}"
        )
    return CLIP_PATH_MAP[clip_variant]
