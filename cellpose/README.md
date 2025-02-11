# Running cellpose on Hail Batch

This folder has the script and tools to run cellpose on the [Hail Batch service](https://hail.is/docs/batch/service.html) to identify cell centers. By default, it uses one g2-standard-4 GPU-enabled machine for each plate, and runs four instances of the script in parallel to process that plate. The output is consolidated into a single tsv file.

## How to use

To get started, copy `config-sample.yaml` to `config.yaml` and fill in the blanks. There are two scripts. The script `run_batch.py` should be run locally to generate and launch the Hail Batch jobs. You will need to provide the path to the input and output folder and plate names. The script `run_cellpose.py` is run by Hail Batch and should be copied to a storage bucket where your Batch service account can pull from.

## Output format
The output is a tsv file with one line per file containing the filename, and x/y coordinates of cell centers:

```
file	x	y
r01c01f01p01-ch2sk1fk1fl1.jxl	[156, 468]	[8, 11]
r01c01f09p01-ch2sk1fk1fl1.jxl	[291, 953, 527, 175]	[11, 21, 42, 46]
```

This file can be read with Pandas via:

```
import pandas as pd
from ast import literal_eval
df = pd.read_csv('file.tsv', sep='\t', converters={'x':literal_eval, 'y':literal_eval})
```

  
## Docker image

The script requires a Docker container with moreutils and cellpose preinstalled. We provide a sample Dockerfile which additionally has JPEG-XL support and nvidia-container-toolkit for GPU-accelerated computing. You could build the image by running the following commands on a Debian VM on GCP in the same folder as Dockerfile. You should replace PROJECT and REPOSITORY to point to your own [artifact repository](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images).

```
sudo apt update
sudo apt install docker.io
sudo usermod -a -G docker $USER
sudo su - $USER
docker build -t pillow-jxl-cuda .
docker tag pillow-jxl-cuda us-central1-docker.pkg.dev/PROJECT/REPOSITORY/pillow-jxl-cuda
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/PROJECT/REPOSITORY/pillow-jxl-cuda
```
