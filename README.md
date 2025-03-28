# Microscopy computational tools

This repository contains scripts to identify cell centers and obtain cell embeddings with pretrained machine learning models. The scripts can be run locally or on the [Hail Batch service](https://hail.is/docs/batch/service.html).

The scripts are easy to modify, e.g., to add a new model.

## How to use

Start by using cellpose to identify cell centers. The output of the cellpose script can be used as input for the script that generates the embeddings.

## Docker image

The Dockerfile in this folder can be used to create a Docker container with the scripts in this repository and all required dependencies. It supports both TIFF and JPEG-XL images. You could build the image by running the following commands on a Debian VM on GCP in the same folder as Dockerfile. You should replace PROJECT and REPOSITORY to point to your own [artifact repository](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images).

```
sudo apt update
sudo apt install docker.io
sudo usermod -a -G docker $USER
sudo su - $USER
docker build -t microscopy_computational_tools .
docker tag microscopy_computational_tools us-central1-docker.pkg.dev/PROJECT/REPOSITORY/microscopy_computational_tools
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/PROJECT/REPOSITORY/microscopy_computational_tools
```

## Running locally on macOS

For quick local testing on macOS, you can use the provided example script:

```bash
# Install dependencies
uv sync

# Run the example script
./example_run.sh
```

Note that:
- All Python scripts must be run with `uv run` or by activating the virtual environment
- The package versions in `pyproject.toml` may not match what is in the `Dockerfile`

### Monitoring GPU Usage

The repository includes tools to monitor GPU usage on macOS:

```bash
# Check GPU availability and run diagnostics
uv run ./tools/check_gpu.py   # Check PyTorch GPU support
./tools/check_macos_gpu.sh    # Check macOS GPU capabilities

# Visual monitoring
./tools/open_activity_monitor.sh  # Open Activity Monitor with GPU tab selected

# Live monitoring during script execution (in a separate terminal)
sudo powermetrics --samplers gpu_power -i 1000  # Sample every 1s
```

The main scripts (cellpose and embeddings) have been instrumented to show which device (CPU/GPU) they're using. Look for messages like:
- "Cellpose model running on device: ..."
- "Using MPS (Metal) device" or "Using CPU device"
- "TensorFlow GPU devices available: ..."

You can see GPU activity in Activity Monitor's GPU tab or by using the `powermetrics` command.
