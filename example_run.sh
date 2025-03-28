#!/bin/bash

# Create directories
mkdir -p images output

# Check system type
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Running on macOS"
    # For macOS, use Activity Monitor for visual monitoring
    echo "Tip: Use Activity Monitor (GPU tab) to visually monitor GPU usage"
    # Check if Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "Apple Silicon detected"
    else
        echo "Intel Mac detected"
    fi
    # Run our GPU check script
    echo "Running PyTorch GPU check script..."
    uv run tools/check_gpu.py
else
    # For non-macOS systems, use nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected"
        nvidia-smi
    else
        echo "No NVIDIA GPU detected or nvidia-smi not installed"
    fi
fi

# Function to download a single image file
download_image() {
  local field=$1
  local channel=$2
  local filename="r01c01f0${field}p01-ch${channel}sk1fk1fl1.tiff"
  local target_path="images/$filename"
  
  # Check if file already exists
  if [ -f "$target_path" ]; then
    echo "File $filename already exists, skipping download"
  else
    echo "Downloading $filename"
    aws s3 cp "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/BR00116991__2020-11-05T19_51_35-Measurement1/Images/$filename" "$target_path"
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

# Create GPU debug wrapper for macOS
if [[ "$(uname)" == "Darwin" ]]; then
    # Function to monitor GPU usage on macOS
    run_with_gpu_monitoring() {
        echo "=== GPU usage before task ==="
        top -l 1 -s 0 | grep "GPU\|Metal"
        
        echo "=== Starting task: $1 ==="
        
        # Set environment variables to force PyTorch on MPS (Metal)
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        
        # Run the command
        "${@:2}"
        
        echo "=== GPU usage after task ==="
        top -l 1 -s 0 | grep "GPU\|Metal"
        
        # Check PyTorch device being used
        uv run tools/check_gpu.py
    }
else
    # For non-macOS systems
    run_with_gpu_monitoring() {
        echo "=== GPU usage before task ==="
        nvidia-smi
        
        echo "=== Starting task: $1 ==="
        "${@:2}"
        
        echo "=== GPU usage after task ==="
        nvidia-smi
    }
fi

# Step 2: Run cellpose in parallel with monitoring
echo "Running cellpose..."
run_with_gpu_monitoring "cellpose" uv run cellpose/run_cellpose.py ./images/ ch5 1 0

mv cellpose_0_1.csv output/cellpose_0_1.csv

# Step 3: Run embeddings with monitoring
echo "Generating embeddings..."
run_with_gpu_monitoring "embeddings" uv run embeddings/run_model.py \
    cpcnn models/Cell_Painting_CNN_v1.hdf5 \
    ./images/ \
    DNA,RNA,AGP,ER,Mito \
    -ch5,-ch3,-ch1,-ch4,-ch2 \
    output/cellpose_0_1.csv \
    0 \
    output/embedding.tsv \
    output/crops.png

echo "All processing completed!" 