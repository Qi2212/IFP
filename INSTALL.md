## Installation

### Requirements
- Linux with Python ≥ 3.10
- PyTorch ≥ 2.8.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together to ensure compatibility:
  ```bash
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  ```
- **SAM**: follow [Segment Anything installation instructions](https://github.com/facebookresearch/segment-anything).
- **SAM 2**: follow [SAM 2 installation instructions](https://github.com/facebookresearch/sam2).
- **DINOv2**: follow [DINOv2 installation instructions](https://github.com/facebookresearch/dinov2).
- **DINOv3**: follow [DINOv3 installation instructions](https://github.com/facebookresearch/dinov3).
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name ifp python=3.10
conda activate ifp

# Install PyTorch with CUDA support
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Clone SAM
git clone https://github.com/facebookresearch/segment-anything.git

# Clone SAM 2
git clone https://github.com/facebookresearch/sam2.git


# Clone DINOv2
git clone https://github.com/facebookresearch/dinov2.git


# Clone and install this project
git clone https://github.com/L-AILab/IFP.git
cd IFP
pip install -r requirements.txt
```

### DINOv3 Weights
DINOv3 requires pre-trained weights. Download them from the [DINOv3 repository](https://github.com/facebookresearch/dinov3) and place the `.pth` file in `dinov3/download_models/`.

### DINOv2 Weights
If using DINOv2, download weights from the [DINOv2 repository](https://github.com/facebookresearch/dinov2) and place them in `dinov2/download_models/`.

### CLIP Weights
Download CLIP weights from [HuggingFace](https://huggingface.co/openai) into `clip/`. Supported variants: `clip-vit-base-patch16`, `clip-vit-large-patch14`.

### SAM / SAM2 Checkpoints
Download SAM or SAM2 checkpoints following their official instructions. Place SAM2 checkpoints under `sam2/checkpoints/`.

### Dataset Preparation
Organize datasets following the structure described in `datasets/`. Use our download script:
```bash
bash scripts/download_datasets.sh
```
