#!/bin/bash

# Create directories
mkdir -p images output

# Function to download a single image file
download_image() {
  local field=$1
  local channel=$2
  local filename="r01c01f0${field}p01-ch${channel}sk1fk1fl1.tiff"
  
  if [ -f "images/$filename" ]; then
    echo "File $filename already exists, skipping download"
  else
    echo "Downloading $filename"
    aws s3 cp "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/BR00116991__2020-11-05T19_51_35-Measurement1/Images/$filename" "images/$filename"
  fi
}

# Download the model file
mkdir -p models
if [ ! -f "models/Cell_Painting_CNN_v1.hdf5" ]; then
  echo "Downloading model file..."
  curl -L "https://zenodo.org/records/7114558/files/Cell_Painting_CNN_v1.hdf5?download=1" -o "models/Cell_Painting_CNN_v1.hdf5"
fi

# Export functions for GNU parallel
export -f download_image

# Step 1: Download all images in parallel
echo "Downloading images..."
parallel download_image {1} {2} ::: $(seq -w 1 4) ::: $(seq 1 5)

# Step 2: Run cellpose in parallel
echo "Running cellpose..."
uv run \
    cellpose/run_cellpose.py \
    ./images/ \
    ch5 \
    1 0

mv cellpose_0_1.csv output/cellpose_0_1.csv

# Step 4: Run embeddings
echo "Generating embeddings..."
uv run embeddings/run_model.py \
    cpcnn models/Cell_Painting_CNN_v1.hdf5 \
    ./images/ \
    DNA,RNA,AGP,ER,Mito \
    -ch5,-ch3,-ch1,-ch4,-ch2 \
    output/cellpose_0_1.csv \
    0 \
    output/embedding.tsv \
    output/crops.png

echo "All processing completed!" 