#!/usr/bin/env bash
# Download a small, real-photo dataset for local pipeline smoke-testing.
#
# Default: COLMAP's own "south-building" sample dataset (233.9 MB, 128 images,
# hosted directly by the COLMAP project) -- small enough to download locally
# on a machine with no GPU, real photos (not synthetic renders), and exercises
# exactly the feature_extractor -> matcher -> mapper path core/sfm.py drives.
#
# For a real object-centric "walking around an object" scene closer to the
# eventual use case, see docs/datasets.md for the (much larger, ~12 GB)
# Mip-NeRF 360 dataset -- better suited to a RunPod pod than a local download.
set -euo pipefail

SCENE="${1:-south-building}"
BASE_URL="https://github.com/colmap/colmap/releases/download/3.11.1"
OUT_DIR="data/raw/${SCENE}"

case "$SCENE" in
  south-building|gerrard-hall) ;;
  *) echo "Unknown scene '$SCENE'. Supported: south-building, gerrard-hall" >&2; exit 1 ;;
esac

if [ -d "$OUT_DIR" ]; then
  echo "Already present at $OUT_DIR -- skipping download."
  exit 0
fi

mkdir -p "$OUT_DIR"
echo "Downloading ${SCENE}.zip..."
curl -L -o "/tmp/${SCENE}.zip" "${BASE_URL}/${SCENE}.zip"

echo "Extracting to ${OUT_DIR}..."
unzip -q "/tmp/${SCENE}.zip" -d "$OUT_DIR"
rm "/tmp/${SCENE}.zip"

# COLMAP's zips nest images/ one level down (<scene>/images/); flatten so
# reconstruct.py's --input can point straight at a directory of photos.
if [ -d "${OUT_DIR}/${SCENE}/images" ]; then
  mv "${OUT_DIR}/${SCENE}/images" "${OUT_DIR}/images_tmp"
  rm -rf "${OUT_DIR:?}/${SCENE}"
  mv "${OUT_DIR}/images_tmp" "${OUT_DIR}/images"
fi

n=$(find "${OUT_DIR}/images" -type f | wc -l)
echo "Done -- ${n} images at ${OUT_DIR}/images"
echo "Run: python reconstruct.py --input ${OUT_DIR}/images --output output/${SCENE} --skip-splat"
