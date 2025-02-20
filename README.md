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