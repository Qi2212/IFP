"""
Loss Functions and Model Components
====================================
ProjectionHead, ClipContrastiveLoss, and ImageFeatureFilesDataset
for the IFP training pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ProjectionHead(nn.Module):
    """MLP projection head with optional hidden layer, output L2-normalized."""

    def __init__(self, in_dim: int, out_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)


class ClipContrastiveLoss(nn.Module):
    """
    Symmetric CLIP-style contrastive loss (InfoNCE).
    Computes text→image and image→text cross-entropy with temperature.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(
        self,
        text_proj: torch.Tensor,
        img_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_proj: [B, D] L2-normalized text projections.
            img_proj: [B, D] L2-normalized image projections.

        Returns:
            Scalar loss tensor.
        """
        logits = torch.matmul(text_proj, img_proj.T) / self.t
        labels = torch.arange(len(text_proj), device=logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_t2i + loss_i2t)


class ImageFeatureFilesDataset(Dataset):
    """
    Dataset that loads pre-extracted image .pt feature files.
    Each .pt file contains:
        {
            "img_path": str,
            "patch_feats": Tensor [N, D],
            "gh": int, "gw": int,
            "objects": [ {classTitle, patch_indices, cluster_centers, k, is_bg, pooled_feat}, ... ]
        }
    """

    def __init__(self, feat_dir: str):
        if not os.path.exists(feat_dir):
            raise FileNotFoundError(f"Feature directory not found: {feat_dir}")

        self.files = sorted([
            os.path.join(feat_dir, f)
            for f in os.listdir(feat_dir)
            if f.endswith(".pt")
        ])

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in: {feat_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], weights_only=False)
        return data


def count_params(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_inter_loss(img_proj: torch.Tensor, background_flags: list, device: str = "cuda:0") -> torch.Tensor:
    """
    Inter-class separation loss: push foreground class centers apart.

    Args:
        img_proj: [B, D] projected image features.
        background_flags: list of bool, True = background.
        device: Device string.

    Returns:
        Scalar loss.
    """
    mask_fg = torch.tensor([not b for b in background_flags], device=device)

    if mask_fg.sum() < 2:
        return torch.tensor(0.0, device=device)

    fg_pooled = img_proj[mask_fg]
    sim_mat = fg_pooled @ fg_pooled.T
    nfg = fg_pooled.size(0)
    mask_offdiag = ~torch.eye(nfg, dtype=torch.bool, device=device)
    return sim_mat[mask_offdiag].mean()


def compute_bg_loss(img_proj: torch.Tensor, background_flags: list, device: str = "cuda:0") -> torch.Tensor:
    """
    Foreground-background separation loss: push FG away from BG.

    Args:
        img_proj: [B, D] projected image features.
        background_flags: list of bool, True = background.
        device: Device string.

    Returns:
        Scalar loss.
    """
    bg_indices = [i for i, b in enumerate(background_flags) if b]
    fg_indices = [i for i, b in enumerate(background_flags) if not b]

    if len(bg_indices) == 0 or len(fg_indices) == 0:
        return torch.tensor(0.0, device=device)

    bg_embs = img_proj[bg_indices]
    fg_embs = img_proj[fg_indices]
    return (fg_embs @ bg_embs.T).mean()


def compute_intra_loss(
    img_head: nn.Module,
    img_proj: torch.Tensor,
    centers_feat_list: list,
    background_flags: list,
    device: str = "cuda:0",
) -> torch.Tensor:
    """
    Intra-class consistency loss: pull cluster centers toward class center.

    Args:
        img_head: Image projection head.
        img_proj: [B, D] projected image features.
        centers_feat_list: list of cluster center arrays (numpy) or None.
        background_flags: list of bool.
        device: Device string.

    Returns:
        Scalar loss.
    """
    import numpy as np
    per_class_losses = []

    for i_cent, centers_np in enumerate(centers_feat_list):
        if background_flags[i_cent]:
            continue
        if centers_np is None:
            continue
        try:
            centers_t = torch.from_numpy(
                np.asarray(centers_np, dtype=np.float32)
            ).to(device)

            centers_proj = img_head(centers_t)  # [K, D]
            pooled_proj = img_proj[i_cent].unsqueeze(1)  # [D, 1]
            sims = (centers_proj @ pooled_proj).squeeze(1)  # [K]
            per_class_losses.append((1.0 - sims).mean())
        except Exception:
            continue

    if len(per_class_losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(per_class_losses).mean()
