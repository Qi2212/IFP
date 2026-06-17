#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py
=============
Single-Image Inference Script — IFP Pipeline (Step 3)

Performs One-Shot Segmentation using a trained DSFA projection head,
DINO, CLIP, and SAM/SAM2. Selects top-K most similar patch features
and feeds them as multi-point prompts to SAM in a single forward pass.

Usage:
    # Single image inference
    python inference.py --config configs/infer_config.yaml \
        --image ./datasets/VOC/person/target_images/bird.jpg \
        --text "bird" \
        --output ./output/

    # Batch test on a dataset
    python inference.py --config configs/infer_config.yaml \
        --dataset_root ./datasets/VOC

The config file specifies:
  - dino: model variant, root, weights_dir
  - clip: variant, local path
  - sam: family (sam1/sam2), variant, checkpoint, config
  - projection: weights path, projection dimension
  - inference: resize, sim_threshold, max_points, device
  - io: image_path, mask_save_dir, dataset_root
"""

import os
import sys
import csv
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_yaml_config
from utils.dino_loader import load_dino_model, extract_tokens_from_dino
from utils.clip_loader import load_clip_model
from utils.mask_utils import compute_iou_and_dice
from utils.data_io import find_gt_mask_path
from utils.losses import ProjectionHead
from utils.model_check import (
    check_path_exists, check_model_loaded, check_device_available,
    ensure_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IFP One-Shot Segmentation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/infer_config.yaml)"
    )
    parser.add_argument(
        "--image", type=str, default="",
        help="Path to a single input image (overrides config io.image_path)"
    )
    parser.add_argument(
        "--text", type=str, default="",
        help="Text prompt for segmentation (overrides config inference.text_prompt)"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output directory for segmentation masks (overrides config io.mask_save_dir)"
    )
    parser.add_argument(
        "--dataset_root", type=str, default="",
        help="Dataset root for batch testing (overrides config io.dataset_root)"
    )
    parser.add_argument(
        "--gt_mask", type=str, default="",
        help="Optional ground-truth mask for IoU/Dice evaluation"
    )
    return parser.parse_args()


def build_sam_predictor(sam_cfg: dict, device: str):
    """
    Build SAM or SAM2 predictor based on config.

    Args:
        sam_cfg: Config dict with family, variant, checkpoint, config keys.
        device: Device string.

    Returns:
        Predictor object with .set_image(np_array) and .predict(...) methods.
    """
    family = sam_cfg.get("family", "sam2")
    variant = sam_cfg.get("variant", "l")

    if family == "sam1":
        from segment_anything import sam_model_registry, SamPredictor

        sam_models = {
            "h": ("vit_h", f"{PROJECT_ROOT}/segment-anything/model/sam_vit_h_4b8939.pth"),
            "l": ("vit_l", f"{PROJECT_ROOT}/segment-anything/model/sam_vit_l_0b3195.pth"),
        }

        if variant not in sam_models:
            raise ValueError(f"Unsupported SAM1 variant: {variant}")

        model_type, checkpoint = sam_models[variant]
        checkpoint = sam_cfg.get("checkpoint", checkpoint)

        sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        sam_model.to(device)
        predictor = SamPredictor(sam_model)
        return predictor

    elif family == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_config = sam_cfg.get("config", "./sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
        sam_checkpoint = sam_cfg.get("checkpoint",
            f"{PROJECT_ROOT}/sam2/checkpoints/sam2.1_hiera_large.pt")

        check_path_exists(sam_config, "SAM2 config file")
        check_path_exists(sam_checkpoint, "SAM2 checkpoint")

        sam2_model = build_sam2(sam_config, sam_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        return predictor

    else:
        raise ValueError(f"Unsupported SAM family: {family}")


def one_shot_segmentation(
    pil_img: Image.Image,
    text_prompt: str,
    dino_model: nn.Module,
    clip_model,
    clip_processor,
    proj_img: ProjectionHead,
    proj_txt: ProjectionHead,
    predictor,
    dino_variant: str = "dino3h",
    resize: int = 512,
    sim_threshold: float = 0.2,
    max_points: int = 15,
    device: str = "cuda:0",
    gt_mask=None,
) -> tuple:
    """
    One-Shot Segmentation: select top-K patches most similar to the text prompt,
    feed them as multi-point prompts to SAM in a single forward pass.

    Args:
        pil_img: Input PIL Image.
        text_prompt: Text instruction/prompt for segmentation.
        dino_model: Loaded DINO model.
        clip_model: Loaded CLIP model.
        clip_processor: CLIP processor.
        proj_img: Trained image projection head.
        proj_txt: Trained text projection head.
        predictor: SAM/SAM2 predictor with .set_image() and .predict().
        dino_variant: 'dino3h', 'dino3l', 'dino2l', 'dino2g', etc.
        resize: Image resize dimension.
        sim_threshold: Minimum cosine similarity for a patch to be considered.
        max_points: Maximum number of point prompts (top-K).
        device: Device string.
        gt_mask: Optional ground-truth PIL mask for IoU/Dice evaluation.

    Returns:
        tuple: (binary_mask_PIL, iou, dice)
    """
    resize_tuple = (resize, resize)

    # Step 1: Extract DINO patch tokens and project
    tokens_np, (h, w) = extract_tokens_from_dino(
        dino_model, pil_img,
        dino_variant=dino_variant,
        resize=resize_tuple,
        device=device,
    )
    toks_proj = proj_img(torch.from_numpy(tokens_np).to(device).float())  # [N, proj_dim]

    # Step 2: Compute text feature and project
    text_inputs = clip_processor(
        text=[text_prompt],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        text_out = clip_model.get_text_features(**text_inputs)
        if not isinstance(text_out, torch.Tensor):
            text_out = text_out.pooler_output
        t_feats = F.normalize(text_out, dim=-1)

    text_proj = proj_txt(t_feats).squeeze(0)  # [proj_dim]

    # Step 3: Compute similarity and select top-K valid patches
    sims = F.cosine_similarity(toks_proj, text_proj.unsqueeze(0), dim=-1)

    valid_idxs = torch.where(sims > sim_threshold)[0]

    ow, oh = pil_img.size

    if valid_idxs.numel() == 0:
        black_mask = Image.fromarray(np.zeros((oh, ow), dtype=np.uint8))
        if gt_mask is None:
            return black_mask, 0.0, 0.0
        return black_mask, *compute_iou_and_dice(black_mask, gt_mask)

    # Sort by similarity descending, take top-K
    valid_sims = sims[valid_idxs]
    sorted_order = torch.argsort(valid_sims, descending=True)
    topk_order = sorted_order[:max_points]
    fg_idxs = valid_idxs[topk_order]

    # Step 4: Set SAM image
    image_np = np.array(pil_img.convert("RGB"))
    predictor.set_image(image_np)

    # Step 5: Build multi-point prompts from top-K patches
    point_coords = []
    for idx in fg_idxs.tolist():
        py, px = divmod(idx, w)
        sx = (px + 0.5) * ow / w
        sy = (py + 0.5) * oh / h
        point_coords.append([sx, sy])

    point_coords = np.array(point_coords, dtype=np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)

    # Step 6: Single SAM forward pass
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        black_mask = Image.fromarray(np.zeros((oh, ow), dtype=np.uint8))
        if gt_mask is None:
            return black_mask, 0.0, 0.0
        return black_mask, *compute_iou_and_dice(black_mask, gt_mask)

    # Step 7: Select best mask by score
    final_mask = masks[np.argmax(scores)].astype(np.uint8)
    pred_img = Image.fromarray(final_mask * 255)

    if gt_mask is None:
        return pred_img, 0.0, 0.0
    return pred_img, *compute_iou_and_dice(pred_img, gt_mask)


def test_single_image(
    image_path: str,
    text_prompt: str,
    cfg: dict,
    dino_model, clip_model, clip_processor,
    proj_img, proj_txt, predictor,
    device: str,
    gt_mask_path: str = "",
) -> dict:
    """Run one-shot segmentation on a single image."""
    dino_cfg = cfg["dino"]
    infer_cfg = cfg["inference"]

    pil_img = Image.open(image_path).convert("RGB")

    gt_mask = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = Image.open(gt_mask_path).convert("L").resize(pil_img.size)

    resize = infer_cfg.get("resize", 512)
    sim_threshold = infer_cfg.get("sim_threshold", 0.2)
    max_points = infer_cfg.get("max_points", 12)

    pred_mask, iou, dice = one_shot_segmentation(
        pil_img=pil_img,
        text_prompt=text_prompt,
        dino_model=dino_model,
        clip_model=clip_model,
        clip_processor=clip_processor,
        proj_img=proj_img,
        proj_txt=proj_txt,
        predictor=predictor,
        dino_variant=dino_cfg.get("variant", "dino3h"),
        resize=resize,
        sim_threshold=sim_threshold,
        max_points=max_points,
        device=device,
        gt_mask=gt_mask,
    )

    return {
        "image": image_path,
        "text": text_prompt,
        "iou": iou,
        "dice": dice,
        "pred_mask": pred_mask,
    }


def test_dataset(
    dataset_root: str,
    text_prompt: str,
    cfg: dict,
    dino_model, clip_model, clip_processor,
    proj_img, proj_txt, predictor,
    device: str,
    mask_save_dir: str,
) -> None:
    """
    Batch test on a dataset organized as:
        <dataset_root>/<class>/target_images/  +  target_masks/
    """
    infer_cfg = cfg["inference"]

    results = []
    summary_results = []

    categories = [
        d for d in sorted(os.listdir(dataset_root))
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    for cat in categories:
        img_dir = os.path.join(dataset_root, cat, "target_images")
        mask_gt_dir = os.path.join(dataset_root, cat, "target_masks")

        if not os.path.exists(img_dir):
            continue

        cat_save_dir = os.path.join(mask_save_dir, cat)
        ensure_dir(cat_save_dir)

        ious, dices = [], []

        for fname in tqdm(sorted(os.listdir(img_dir)), desc=f"[Test] {cat}"):
            if fname.startswith("."):
                continue

            img_path = os.path.join(img_dir, fname)
            name, _ = os.path.splitext(fname)

            gt_mask_path = ""
            if os.path.exists(mask_gt_dir):
                gt_mask_path = find_gt_mask_path(mask_gt_dir, name) or ""

            # Use category name as prompt, or override with provided text
            prompt = text_prompt if text_prompt else cat

            result = test_single_image(
                image_path=img_path,
                text_prompt=prompt,
                cfg=cfg,
                dino_model=dino_model,
                clip_model=clip_model,
                clip_processor=clip_processor,
                proj_img=proj_img,
                proj_txt=proj_txt,
                predictor=predictor,
                device=device,
                gt_mask_path=gt_mask_path,
            )

            # Save mask
            result["pred_mask"].save(os.path.join(cat_save_dir, f"{name}.png"))

            results.append([img_path, cat, result["iou"], result["dice"]])
            ious.append(result["iou"])
            dices.append(result["dice"])

        if len(ious) > 0:
            summary_results.append([cat, np.mean(dices), np.mean(ious)])
            print(f"[{cat}] mIoU={np.mean(ious):.4f}, mDice={np.mean(dices):.4f}")

    # Save CSVs
    csv_file = os.path.join(mask_save_dir, "results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "category", "iou", "dice"])
        writer.writerows(results)

    summary_csv = os.path.join(mask_save_dir, "results_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "dice", "iou"])
        writer.writerows(summary_results)

    print(f"\nResults saved to: {csv_file}")
    print(f"Summary saved to: {summary_csv}")


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # ---- Validate config ----
    for section in ["dino", "clip", "sam", "projection", "inference"]:
        if section not in cfg:
            raise KeyError(f"Missing config section: [{section}]")

    dino_cfg = cfg["dino"]
    clip_cfg = cfg["clip"]
    sam_cfg = cfg["sam"]
    proj_cfg = cfg["projection"]
    infer_cfg = cfg["inference"]
    io_cfg = cfg.get("io", {})

    # ---- Setup ----
    device_str = infer_cfg.get("device", "cuda:0")
    device = check_device_available(device_str)

    image_path = args.image or io_cfg.get("image_path", "")
    text_prompt = args.text or infer_cfg.get("text_prompt", "")
    output_dir = args.output or io_cfg.get("mask_save_dir", "./output")
    dataset_root = args.dataset_root or io_cfg.get("dataset_root", "")

    ensure_dir(output_dir)

    print(f"=== IFP One-Shot Inference ===")
    print(f"  DINO variant:   {dino_cfg.get('variant', 'dino3h')}")
    print(f"  CLIP variant:   {clip_cfg.get('variant', 'vitb')}")
    print(f"  SAM:            {sam_cfg.get('family', 'sam2')}-{sam_cfg.get('variant', 'l')}")
    print(f"  Projection:     {proj_cfg.get('weights_path', '')}")
    print(f"  Resize:         {infer_cfg.get('resize', 512)}")
    print(f"  Sim threshold:  {infer_cfg.get('sim_threshold', 0.2)}")
    print(f"  Max points:     {infer_cfg.get('max_points', 12)}")
    print(f"  Device:         {device}")

    # ---- Load DINO ----
    print("\n[1/4] Loading DINO model...")
    dino_model = load_dino_model(
        dino_variant=dino_cfg.get("variant", "dino3h"),
        dino_root=dino_cfg.get("root", "./dinov3"),
        weights_dir=dino_cfg.get("weights_dir", "./dinov3/download_models"),
        device=str(device),
    )
    check_model_loaded(dino_model, "DINO")

    # ---- Load CLIP ----
    print("\n[2/4] Loading CLIP model...")
    clip_model, clip_processor = load_clip_model(
        clip_path=clip_cfg.get("path", "./clip/clip-vit-base-patch16"),
        device=str(device),
    )
    check_model_loaded(clip_model, "CLIP")

    # ---- Load projection heads ----
    print("\n[3/4] Loading projection heads...")
    proj_dim = proj_cfg.get("proj_dim", 512)
    dino_dim = dino_model.embed_dim
    clip_proj_dim = clip_model.config.projection_dim

    proj_img = ProjectionHead(dino_dim, proj_dim).to(device)
    proj_txt = ProjectionHead(clip_proj_dim, proj_dim).to(device)

    weights_path = proj_cfg.get("weights_path", "./weights/pretrain_coco_dino3h_clipb_16.pth")
    check_path_exists(weights_path, "Projection weights")

    ckpt = torch.load(weights_path, map_location=str(device))

    # Support multiple checkpoint formats (from train.py output)
    if "img_head" in ckpt:
        proj_img.load_state_dict(ckpt["img_head"])
        proj_txt.load_state_dict(ckpt["txt_head"])
    elif "img" in ckpt:
        proj_img.load_state_dict(ckpt["img"])
        proj_txt.load_state_dict(ckpt["txt"])
    else:
        raise KeyError(
            "Checkpoint format not recognized. "
            "Expected keys: 'img_head'/'txt_head' or 'img'/'txt'"
        )

    proj_img.eval()
    proj_txt.eval()
    print(f"  Loaded weights from: {weights_path}")

    # ---- Build SAM predictor ----
    print("\n[4/4] Loading SAM predictor...")
    predictor = build_sam_predictor(sam_cfg, str(device))
    print(f"  SAM {sam_cfg.get('family')}-{sam_cfg.get('variant')} loaded.")

    # ---- Run inference ----
    print("\n=== Running One-Shot Inference ===")

    if dataset_root:
        # Batch test mode
        print(f"  Dataset root: {dataset_root}")
        test_dataset(
            dataset_root=dataset_root,
            text_prompt=text_prompt,
            cfg=cfg,
            dino_model=dino_model,
            clip_model=clip_model,
            clip_processor=clip_processor,
            proj_img=proj_img,
            proj_txt=proj_txt,
            predictor=predictor,
            device=str(device),
            mask_save_dir=output_dir,
        )

    elif image_path:
        # Single image mode
        check_path_exists(image_path, "Input image")
        print(f"  Image:       {image_path}")
        print(f"  Text:        {text_prompt}")
        print(f"  Output dir:  {output_dir}")

        result = test_single_image(
            image_path=image_path,
            text_prompt=text_prompt,
            cfg=cfg,
            dino_model=dino_model,
            clip_model=clip_model,
            clip_processor=clip_processor,
            proj_img=proj_img,
            proj_txt=proj_txt,
            predictor=predictor,
            device=str(device),
            gt_mask_path=args.gt_mask,
        )

        # Save mask
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        result["pred_mask"].save(mask_path)

        print(f"\n  Mask saved to: {mask_path}")
        print(f"  IoU:  {result['iou']:.4f}")
        print(f"  Dice: {result['dice']:.4f}")

    else:
        print(
            "  [warn] No image or dataset specified. "
            "Use --image <path> for single image or --dataset_root <path> for batch mode."
        )

    print("\n=== Inference complete ===")


if __name__ == "__main__":
    main()
