#!/bin/bash
# ============================================================
# Download Datasets Script for IFP Project
# ============================================================
# This script downloads datasets referenced in the IFP paper.
# Datasets: COCO, VOC, ISIC, Kvasir, GBMSD
# ============================================================

set -e

DATASET_DIR="$(cd "$(dirname "$0")/../datasets" && pwd)"

echo "=== Downloading datasets to ${DATASET_DIR} ==="

# ---- COCO-Stuff ----
echo "[1/5] Download COCO-Stuff..."
# COCO 2017 images
# wget http://images.cocodataset.org/zips/train2017.zip
# wget http://images.cocodataset.org/zips/val2017.zip
# COCO-Stuff annotations available at: https://github.com/nightrome/cocostuff
echo "  -> Please download COCO images manually from https://cocodataset.org/"
echo "  -> See ${DATASET_DIR}/COCO/ for organization"

# ---- Pascal VOC ----
echo "[2/5] Download Pascal VOC..."
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "  -> Please download VOC from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
echo "  -> See ${DATASET_DIR}/VOC/ for organization"

# ---- ISIC ----
echo "[3/5] Download ISIC..."
echo "  -> Please download ISIC 2018 from https://challenge.isic-archive.com/data/"
echo "  -> See ${DATASET_DIR}/ISIC/ for organization"

# ---- Kvasir ----
echo "[4/5] Download Kvasir..."
echo "  -> Please download Kvasir-SEG from https://datasets.simula.no/kvasir-seg/"
echo "  -> See ${DATASET_DIR}/Kvasir/ for organization"

# ---- GBMSD ----
echo "[5/5] Download GBMSD..."
echo "  -> Please download GBMSD from the official source"
echo "  -> See ${DATASET_DIR}/GBMSD/ for organization"

echo ""
echo "=== Dataset download guide complete ==="
echo "Please organize each dataset as:"
echo "  datasets/<name>/reference_images/"
echo "  datasets/<name>/reference_masks/"
echo "  datasets/<name>/target_images/"
echo "  datasets/<name>/target_masks/"
