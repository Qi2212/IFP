# Preparing Few-Shot Segmentation Datasets

Download following datasets:

## 1. COCO-Stuff
Download COCO2017 train/val images and COCO-Stuff annotations:
```bash
# Download from official sources
# Place images under datasets/COCO/<category>/
# Example: datasets/COCO/person/
```

## 2. Pascal VOC
Download VOC2012 images and annotations:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
Place images under `datasets/VOC/<category>/` (e.g., `datasets/VOC/person/`).

## 3. ISIC
ISIC 2018 skin lesion dataset.
Place images under `datasets/ISIC/ISIC/`.

## 4. Kvasir
Kvasir-SEG gastrointestinal polyp dataset.
Place images under `datasets/Kvasir/Kvasir/`.

## 5. GBMSD
Glomerular Basement Membrane Segmentation Dataset.
Place images under `datasets/GBMSD/GBMSD/`.

---

## Directory Structure

Each dataset follows this layout:

```
datasets/<dataset_name>/
└── <class_name>/
    ├── reference_images/    # Few-shot reference images
    ├── reference_masks/     # Few-shot reference masks
    ├── target_images/       # Test images
    └── target_masks/        # Test ground-truth masks
```

### Multi-class Datasets (VOC, COCO)
Datasets with multiple semantic categories have one subdirectory per class:
```
datasets/VOC/
├── person/
│   ├── reference_images/
│   ├── reference_masks/
│   ├── target_images/
│   └── target_masks/
├── car/
│   ├── ...
├── dog/
│   ├── ...
└── ...
```

### Single-class Datasets (ISIC, Kvasir, GBMSD)
Medical datasets with a single category use the dataset name as the class name:
```
datasets/ISIC/
└── ISIC/
    ├── reference_images/
    ├── reference_masks/
    ├── target_images/
    └── target_masks/
```

- `reference_images/`: Used for few-shot alignment (training phase)
- `reference_masks/`: Ground-truth masks for reference images
- `target_images/`: Images to be segmented (inference phase)
- `target_masks/`: Ground-truth masks for evaluation

See the `.txt` placeholder files in each directory for dataset-specific notes.
