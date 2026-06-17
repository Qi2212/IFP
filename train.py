#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
=========
Training Script — IFP Pipeline (Step 2)

Trains projection heads (image and text) using pre-extracted features.
Uses contrastive alignment loss with inter-class, background, and intra-class
regularization for Dual-Semantic Focus Alignment (DSFA).

Usage:
    python train.py --config configs/train_config.yaml

The config file specifies:
  - data: text/image feature directories
  - model: projection head dimensions
  - train: epochs, batch size, lr, seed, temperature, device
  - loss: lambda weights for each loss term
  - output: weights directory

Output:
  - Best model:  <weights_dir>/<dataset>_<nref>_<dino>_<clip>_best.pth
  - Final model: <weights_dir>/<dataset>_<nref>_<dino>_<clip>_final.pth
"""

import os
import sys
import time
import argparse

import numpy as np
from tqdm import tqdm

import torch
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_yaml_config
from utils.data_io import load_text_feature_map
from utils.losses import (
    ProjectionHead, ClipContrastiveLoss, ImageFeatureFilesDataset,
    count_params, compute_inter_loss, compute_bg_loss, compute_intra_loss,
)
from utils.model_check import (
    ensure_dir, check_device_available, validate_config_keys,
    check_tensor_finite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IFP Training (DSFA Projection Heads)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/train_config.yaml)"
    )
    return parser.parse_args()


def build_weights_filename(cfg: dict) -> str:
    """Build output filename: {dataset}_{nref}_{dino}_{clip}"""
    data_cfg = cfg["data"]
    dino_cfg = cfg.get("dino", {})
    clip_cfg = cfg.get("clip", {})

    dataset = data_cfg.get("dataset_name", "dataset")
    nref = data_cfg.get("num_reference_images", 10)
    dino_var = dino_cfg.get("variant", "dino3h")
    clip_var = clip_cfg.get("variant", "vitb")

    return f"{dataset}_{nref}_{dino_var}_clip{clip_var}"


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # ---- Validate config ----
    for section in ["data", "model", "train", "loss", "output"]:
        if section not in cfg:
            raise KeyError(f"Missing config section: [{section}]")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    output_cfg = cfg["output"]

    # ---- Setup ----
    device = check_device_available(train_cfg.get("device", "cuda:0"))
    seed = train_cfg.get("seed", 0)

    np.random.seed(seed)
    torch.manual_seed(seed)

    EPOCHS = train_cfg.get("epochs", 100)
    BATCH_IMAGES = train_cfg.get("batch_size", 8)
    LR = train_cfg.get("lr", 0.001)
    PROJ_DIM = model_cfg.get("proj_dim", 512)
    HIDDEN_DIM = model_cfg.get("hidden_dim", 512)
    TEMPERATURE = train_cfg.get("temperature", 0.07)

    LAMBDA_INTER = loss_cfg.get("lambda_inter", 0.5)
    LAMBDA_BG = loss_cfg.get("lambda_bg", 0.5)
    LAMBDA_INTRA = loss_cfg.get("lambda_intra", 0.5)

    TEXT_FEATURE_DIR = data_cfg.get("text_feature_dir", "./features/text_features")
    IMG_FEATURE_DIR = data_cfg.get("img_feature_dir", "./features/image_features")

    weights_dir = output_cfg.get("weights_dir", "./weights")
    ensure_dir(weights_dir)

    base_name = build_weights_filename(cfg)

    print(f"=== IFP Training ===")
    print(f"  Text features:  {TEXT_FEATURE_DIR}")
    print(f"  Image features: {IMG_FEATURE_DIR}")
    print(f"  Proj dim:       {PROJ_DIM}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Batch size:     {BATCH_IMAGES}")
    print(f"  LR:             {LR}")
    print(f"  Loss weights:   inter={LAMBDA_INTER} bg={LAMBDA_BG} intra={LAMBDA_INTRA}")
    print(f"  Temperature:    {TEMPERATURE}")
    print(f"  Output:         {weights_dir}/{base_name}_*.pth")
    print(f"  Device:         {device}")

    # ---- Load text features ----
    print("\n[1/4] Loading text features...")
    text_feat_map = load_text_feature_map(TEXT_FEATURE_DIR, device=str(device))

    # ---- Infer dimensions ----
    print("\n[2/4] Inferring feature dimensions...")
    sample_files = sorted([f for f in os.listdir(IMG_FEATURE_DIR) if f.endswith(".pt")])
    if len(sample_files) == 0:
        raise FileNotFoundError(f"No .pt files in {IMG_FEATURE_DIR}")

    sample = torch.load(
        os.path.join(IMG_FEATURE_DIR, sample_files[0]),
        weights_only=False,
    )
    dino_dim = sample["patch_feats"].shape[1]
    some_text = next(iter(text_feat_map.values()))
    text_dim = some_text.shape[0]

    print(f"  DINO feature dim: {dino_dim}")
    print(f"  Text feature dim: {text_dim}")

    # ---- Build models ----
    print("\n[3/4] Building projection heads...")
    img_head = ProjectionHead(dino_dim, PROJ_DIM, HIDDEN_DIM).to(device)
    txt_head = ProjectionHead(text_dim, PROJ_DIM, HIDDEN_DIM).to(device)

    print(f"  Image head params: {count_params(img_head):,}")
    print(f"  Text head params:  {count_params(txt_head):,}")
    print(f"  Total trainable:   {count_params(img_head) + count_params(txt_head):,}")

    optimizer = torch.optim.AdamW(
        list(img_head.parameters()) + list(txt_head.parameters()),
        lr=LR,
    )
    criterion = ClipContrastiveLoss(temperature=TEMPERATURE)

    # ---- DataLoader ----
    dataset = ImageFeatureFilesDataset(IMG_FEATURE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_IMAGES,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )
    print(f"  Training samples: {len(dataset)}")

    # ---- Training loop ----
    print("\n[4/4] Training...")
    best_loss = float("inf")
    best_epoch = 0
    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        nsteps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

        for batch_files in pbar:
            texts_for_proj = []
            img_pooled_list = []
            background_flags = []
            centers_feat_list = []

            # Collect objects from all images in batch
            for img_dict in batch_files:
                objects = img_dict.get("objects", [])
                for obj in objects:
                    cls_name = obj["classTitle"]
                    pooled = np.asarray(obj["pooled_feat"], dtype=np.float32)
                    img_pooled_list.append(pooled)
                    texts_for_proj.append(cls_name)
                    background_flags.append(bool(obj.get("is_bg", False)))
                    centers_feat_list.append(obj.get("cluster_centers", None))

            if len(img_pooled_list) == 0:
                continue

            # Build tensors
            img_embs = torch.from_numpy(
                np.stack(img_pooled_list, axis=0)
            ).float().to(device)

            # Look up text features
            txt_feats_list = []
            for cls in texts_for_proj:
                if cls in text_feat_map:
                    txt_feats_list.append(text_feat_map[cls].cpu().numpy())
                else:
                    # Fallback: try safe name
                    safe = cls.replace("/", "_").replace(" ", "_")
                    if safe in text_feat_map:
                        txt_feats_list.append(text_feat_map[safe].cpu().numpy())
                    else:
                        txt_feats_list.append(
                            np.zeros((text_dim,), dtype=np.float32)
                        )

            txt_feats = torch.from_numpy(
                np.stack(txt_feats_list, axis=0)
            ).float().to(device)

            # Forward
            img_proj = img_head(img_embs)
            txt_proj = txt_head(txt_feats)

            # Compute losses
            clip_loss = criterion(txt_proj, img_proj)
            inter_loss = compute_inter_loss(img_proj, background_flags, str(device))
            bg_loss = compute_bg_loss(img_proj, background_flags, str(device))

            intra_loss = torch.tensor(0.0, device=device)
            if LAMBDA_INTRA > 0.0:
                intra_loss = compute_intra_loss(
                    img_head, img_proj, centers_feat_list,
                    background_flags, str(device),
                )

            total_loss = (
                1.0 * clip_loss +
                LAMBDA_INTER * inter_loss +
                LAMBDA_BG * bg_loss +
                LAMBDA_INTRA * intra_loss
            )

            # Validate loss values
            check_tensor_finite(total_loss, "total_loss")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            nsteps += 1

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "clip": f"{clip_loss.item():.4f}",
            })

        avg_loss = epoch_loss / max(1, nsteps)
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{EPOCHS} — "
            f"avg_loss {avg_loss:.4f} — "
            f"time {epoch_time/60:.2f}m"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "img_head": img_head.state_dict(),
            "txt_head": txt_head.state_dict(),
            "img": img_head.state_dict(),
            "txt": txt_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "config": {
                "proj_dim": PROJ_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dino_dim": dino_dim,
                "text_dim": text_dim,
                "temperature": TEMPERATURE,
                "lambda_inter": LAMBDA_INTER,
                "lambda_bg": LAMBDA_BG,
                "lambda_intra": LAMBDA_INTRA,
            },
        }

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_path = os.path.join(weights_dir, f"{base_name}_best.pth")
            torch.save(ckpt, best_path)
            print(f"  >> new best saved (epoch {best_epoch}, loss={best_loss:.4f})")

    # Save final
    final_path = os.path.join(weights_dir, f"{base_name}_final.pth")
    torch.save(ckpt, final_path)
    print(f"\n  Final model saved to: {final_path}")

    total_time = time.time() - total_start
    print(f"\n=== Training finished ===")
    print(f"  Best epoch:  {best_epoch}")
    print(f"  Best loss:   {best_loss:.4f}")
    print(f"  Total time:  {total_time/60:.1f}m")
    print(f"  Best model:  {os.path.join(weights_dir, f'{base_name}_best.pth')}")
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    main()
