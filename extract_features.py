#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_features.py
====================
Feature Extraction Script — IFP Pipeline (Step 1)

Extracts DINO patch features and CLIP text features from a dataset,
then saves them to disk for downstream training.

Usage:
    python extract_features.py --config configs/extract_config.yaml

The config file specifies:
  - data: dataset name, path to images/annotations
  - dino: model variant, repo root, weights directory
  - clip: model variant, local model path
  - extract: resize, seed, max_k, device
  - output: feature root directory

This script reads annotations from data_root/img/ and data_root/ann/,
extracts DINO patch embeddings, computes CLIP text features for all classes,
performs adaptive K-Means clustering on foreground patches, and saves:
  - text features  -> <feature_root>/text_features/<class>.pt
  - image features -> <feature_root>/image_features/<image_name>.pt
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: add project root to sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_yaml_config
from utils.dino_loader import load_dino_model, get_patch_size
from utils.clip_loader import load_clip_model, compute_text_features
from utils.feature_extraction import extract_patch_embeddings_from_image
from utils.mask_utils import decode_bitmap_data, pixel_mask_to_patch_mask
from utils.data_io import load_annotation_list, collect_unique_classes
from utils.clustering import adaptive_cluster
from utils.model_check import (
    check_path_exists, check_model_loaded, check_device_available,
    ensure_dir, validate_config_keys,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IFP Feature Extraction"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/extract_config.yaml)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # ---- Validate config ----
    for section in ["data", "dino", "clip", "extract", "output"]:
        if section not in cfg:
            raise KeyError(f"Missing config section: [{section}]")

    data_cfg = cfg["data"]
    dino_cfg = cfg["dino"]
    clip_cfg = cfg["clip"]
    extract_cfg = cfg["extract"]
    output_cfg = cfg["output"]

    # ---- Setup ----
    device = check_device_available(extract_cfg.get("device", "cuda:0"))
    resize = extract_cfg.get("resize", 512)
    resize_tuple = (resize, resize)
    seed = extract_cfg.get("seed", 0)
    max_k = extract_cfg.get("max_k", 5)
    text_batch_size = extract_cfg.get("text_batch_size", 32)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Output directories ----
    feature_root = output_cfg.get("feature_root", "./features")
    text_out_dir = os.path.join(feature_root, "text_features")
    img_out_dir = os.path.join(feature_root, "image_features")
    ensure_dir(text_out_dir)
    ensure_dir(img_out_dir)

    dataset_name = data_cfg.get("dataset_name", "dataset")

    print(f"=== IFP Feature Extraction ===")
    print(f"  Dataset:      {dataset_name}")
    print(f"  Data root:    {data_cfg['data_root']}")
    print(f"  DINO variant: {dino_cfg.get('variant', 'dino3h')}")
    print(f"  CLIP variant: {clip_cfg.get('variant', 'vitb')}")
    print(f"  Resize:       {resize_tuple}")
    print(f"  Max K:        {max_k}")
    print(f"  Output:       {feature_root}")
    print(f"  Device:       {device}")

    # ---- Load DINO model ----
    print("\n[1/5] Loading DINO model...")
    dino_model = load_dino_model(
        dino_variant=dino_cfg.get("variant", "dino3h"),
        dino_root=dino_cfg.get("root", "./dinov3"),
        weights_dir=dino_cfg.get("weights_dir", "./dinov3/download_models"),
        device=str(device),
    )
    check_model_loaded(dino_model, "DINO")

    # ---- Load CLIP model ----
    print("\n[2/5] Loading CLIP model...")
    clip_model, clip_processor = load_clip_model(
        clip_path=clip_cfg.get("path", "./clip/clip-vit-base-patch16"),
        device=str(device),
    )
    check_model_loaded(clip_model, "CLIP")

    # ---- Load annotations ----
    print("\n[3/5] Loading annotations...")
    items = load_annotation_list(data_cfg["data_root"])
    print(f"  Found {len(items)} images with annotations.")

    all_classes = collect_unique_classes(items)
    print(f"  Found {len(all_classes)} unique classes.")

    # ---- Compute text features ----
    print("\n[4/5] Computing text features...")
    for i in range(0, len(all_classes), text_batch_size):
        sub = all_classes[i:i + text_batch_size]
        t_feats = compute_text_features(
            clip_model, clip_processor, sub,
            device=str(device), normalize=True,
        ).cpu()

        for cls_name, feat in zip(sub, t_feats):
            safe_name = cls_name.replace("/", "_").replace(" ", "_")
            path = os.path.join(text_out_dir, f"{safe_name}.pt")
            torch.save({"text": cls_name, "text_feat": feat}, path)

    print(f"  Saved {len(all_classes)} text features to {text_out_dir}")

    # ---- Extract image features ----
    print("\n[5/5] Extracting per-image patch features...")
    patch_size = get_patch_size(dino_model)

    for it in tqdm(items, desc="Extracting"):
        img_path = it["img_path"]
        ann_objs = it["ann"]

        if len(ann_objs) == 0:
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] Failed to open {img_path}: {e}")
            continue

        W_orig, H_orig = pil.size

        # Extract patch features
        patch_feats_t, (gh, gw) = extract_patch_embeddings_from_image(
            dino_model, pil,
            resize=resize_tuple,
            device=str(device),
        )

        patch_feats_norm = F.normalize(patch_feats_t, dim=-1).numpy()

        objects_out = []

        for obj in ann_objs:
            bmp = obj.get("bitmap")
            cls_name = obj.get("classTitle", "object")

            if not bmp:
                continue

            mask = decode_bitmap_data(bmp, image_size=(H_orig, W_orig))
            if mask.sum() == 0:
                continue

            patch_mask = pixel_mask_to_patch_mask(
                mask, resize=resize_tuple, patch_size=patch_size,
            )
            idxs = np.where(patch_mask.reshape(-1) == 1)[0]

            if len(idxs) == 0:
                continue

            idxs = np.array(sorted(set(idxs)), dtype=np.int64)

            feats_selected = patch_feats_norm[idxs]  # [M, D]

            # Pooled feature (class-level focus)
            pooled = feats_selected.mean(axis=0)
            pooled = pooled / (np.linalg.norm(pooled) + 1e-12)

            # Adaptive clustering for structural prototypes
            centers, k_use = adaptive_cluster(feats_selected, max_k=max_k, seed=seed)

            # Background flag
            is_bg = (
                ("other" in cls_name.lower()) or
                ("background" in cls_name.lower()) or
                ("unlabeled" in cls_name.lower())
            )

            obj_dict = {
                "classTitle": cls_name,
                "patch_indices": idxs.astype(np.int64),
                "cluster_centers": centers.astype(np.float32) if centers is not None else None,
                "k": k_use,
                "is_bg": bool(is_bg),
                "pooled_feat": pooled.astype(np.float32),
            }
            objects_out.append(obj_dict)

        if len(objects_out) == 0:
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_obj = {
            "img_path": img_path,
            "patch_feats": patch_feats_t,
            "gh": gh,
            "gw": gw,
            "objects": objects_out,
        }
        torch.save(save_obj, os.path.join(img_out_dir, f"{base}.pt"))

    print(f"\n=== Feature extraction complete ===")
    print(f"  Text features:  {text_out_dir} ({len(all_classes)} classes)")
    print(f"  Image features: {img_out_dir}")


if __name__ == "__main__":
    main()
