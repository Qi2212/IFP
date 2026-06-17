"""
IFP Utils Package
=================
Utility modules for the Instruction-Focus-Prompt (IFP) pipeline.

Modules:
  - dino_loader:     Unified DINOv2/DINOv3 model loading
  - clip_loader:     CLIP model loading and text feature extraction
  - feature_extraction: Patch embedding extraction from images
  - mask_utils:      Mask conversion, decoding, and IoU/Dice metrics
  - data_io:         Dataset annotation loading and I/O helpers
  - clustering:      Adaptive K-Means clustering for patch features
  - losses:          ProjectionHead, ClipContrastiveLoss, Dataset, loss functions
  - model_check:     Model validation and robustness checks
  - config_loader:   YAML configuration loading and merging
"""

from .dino_loader import (
    get_patch_size,
    load_dino_model,
    extract_tokens_from_dino,
)

from .clip_loader import (
    load_clip_model,
    compute_text_features,
    get_clip_path_by_variant,
)

from .feature_extraction import (
    extract_patch_embeddings_from_image,
)

from .mask_utils import (
    decode_bitmap_data,
    pixel_mask_to_patch_mask,
    mask_to_patch_mask,
    compute_iou_and_dice,
)

from .data_io import (
    load_annotation_list,
    load_text_feature_map,
    find_gt_mask_path,
    collect_unique_classes,
)

from .clustering import (
    adaptive_cluster,
)

from .losses import (
    ProjectionHead,
    ClipContrastiveLoss,
    ImageFeatureFilesDataset,
    count_params,
    compute_inter_loss,
    compute_bg_loss,
    compute_intra_loss,
)

from .model_check import (
    check_model_loaded,
    validate_config_keys,
    ensure_dir,
    check_path_exists,
    check_device_available,
    check_tensor_finite,
    check_gradient_flow,
)

from .config_loader import (
    load_yaml_config,
    merge_configs,
    resolve_relative_path,
)
