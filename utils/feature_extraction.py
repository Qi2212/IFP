"""
Feature Extraction Utilities
=============================
Patch embedding extraction from images using DINO models.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import Tuple

from .dino_loader import get_patch_size


def extract_patch_embeddings_from_image(
    model,
    pil_img: Image.Image,
    resize: Tuple[int, int] = (512, 512),
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Extract patch-level DINO features from a PIL image.

    Args:
        model: DINO model on target device.
        pil_img: PIL Image.
        resize: (H, W) resize dimensions.
        device: Device string.

    Returns:
        tuple: (patch_feats [N, D] CPU tensor, (gh, gw)) spatial grid.
    """
    img = pil_img.convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    inp = preprocess(img).unsqueeze(0).to(device)

    PATCH_SIZE = get_patch_size(model)
    expect_h = resize[0] // PATCH_SIZE
    expect_w = resize[1] // PATCH_SIZE
    expected_n = expect_h * expect_w

    with torch.no_grad():
        out = model.get_intermediate_layers(inp, n=1)[0]

        if out.shape[1] == expected_n + 1:
            tokens = out[:, 1:, :]
        else:
            tokens = out

        if tokens.shape[1] < expected_n:
            pad = expected_n - tokens.shape[1]
            tokens = torch.cat([
                tokens,
                tokens[:, :pad, :]
            ], dim=1)
        elif tokens.shape[1] > expected_n:
            tokens = tokens[:, :expected_n, :]

        patch_feats = tokens.squeeze(0).cpu()

    return patch_feats, (expect_h, expect_w)
