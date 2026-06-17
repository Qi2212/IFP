"""
Data I/O Utilities
===================
Loading annotations, text features, and dataset file handling.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional


def load_annotation_list(data_root: str) -> List[Dict]:
    """
    Load image-annotation pairs from a directory structured as:
        data_root/
          img/    (image files)
          ann/    (JSON annotation files named <img> + '.json')

    Args:
        data_root: Root directory containing 'img' and 'ann' subdirectories.

    Returns:
        list of dict: [{"img_path": str, "ann": [objects]}, ...]
    """
    img_dir = os.path.join(data_root, "img")
    ann_dir = os.path.join(data_root, "ann")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    items = []

    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, fn)
        ann_path = os.path.join(ann_dir, fn + ".json")

        if not os.path.exists(ann_path):
            # Try without extension
            base = os.path.splitext(fn)[0]
            ann_path_alt = os.path.join(ann_dir, base + ".json")
            if os.path.exists(ann_path_alt):
                ann_path = ann_path_alt
            else:
                continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        objects = ann_data.get("objects", [])
        items.append({
            "img_path": img_path,
            "ann": objects,
        })

    return items


def load_text_feature_map(
    text_feature_dir: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load all text features from a directory of .pt files.

    Each .pt file is expected to contain:
        {"text": str, "text_feat": Tensor [D]}

    Args:
        text_feature_dir: Directory containing text feature .pt files.
        device: Device to place tensors on.

    Returns:
        dict: Mapping from class name (str) to text feature tensor.
    """
    if not os.path.exists(text_feature_dir):
        raise FileNotFoundError(f"Text feature directory not found: {text_feature_dir}")

    text_feat_map = {}

    for fn in sorted(os.listdir(text_feature_dir)):
        if not fn.endswith(".pt"):
            continue

        d = torch.load(
            os.path.join(text_feature_dir, fn),
            weights_only=False,
        )
        txt = d.get("text", d.get("classTitle", fn[:-3]))
        feat = d["text_feat"].float().to(device)

        # Store by both raw text and safe name
        text_feat_map[txt] = feat
        safe_name = txt.replace("/", "_").replace(" ", "_")
        if safe_name != txt:
            text_feat_map[safe_name] = feat

    print(f"Loaded {len(text_feat_map)} text feature entries from {text_feature_dir}")
    return text_feat_map


def find_gt_mask_path(mask_dir: str, base_name: str) -> Optional[str]:
    """
    Find a ground-truth mask file for a given image base name.

    Args:
        mask_dir: Directory to search for mask files.
        base_name: Image filename stem (without extension).

    Returns:
        str or None: Path to the mask file, or None if not found.
    """
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        cand = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(cand):
            return cand
    return None


def collect_unique_classes(items: List[Dict]) -> List[str]:
    """
    Collect all unique class titles from a list of annotation items.

    Args:
        items: List of {"img_path": str, "ann": [objects]} dicts.

    Returns:
        list: Sorted list of unique class title strings.
    """
    all_classes = set()

    for it in items:
        for obj in it.get("ann", []):
            title = obj.get("classTitle")
            if title:
                all_classes.add(title)

    return sorted(list(all_classes))
