FROM 'debian:12-slim'
ARG JXL_VERSION="0.11.1"

RUN apt update && \
    # some tools, moreutils (has parallel), and some Pillow and opencv dependencies
    apt install -y python3 python3-pip git curl moreutils libtiff-dev zlib1g-dev libhwy-dev libgl1 libglib2.0-0 && \
    curl -L --output jxl-linux-x86_64-static.tar.gz https://github.com/libjxl/libjxl/releases/download/v${JXL_VERSION}/jxl-linux-x86_64-static-v${JXL_VERSION}.tar.gz && \
    tar -zxvf jxl-linux-x86_64-static.tar.gz  && \
    mv tools/* /usr/bin && \
    curl -L --output jxl-debs-amd64-debian-bookworm.tar.gz https://github.com/libjxl/libjxl/releases/download/v${JXL_VERSION}/jxl-debs-amd64-debian-bookworm-v${JXL_VERSION}.tar.gz && \
    tar -zxf jxl-debs-amd64-debian-bookworm.tar.gz && \
    apt install -y ./libjxl_${JXL_VERSION}_amd64.deb ./libjxl-dev_${JXL_VERSION}_amd64.deb && \
    rm /usr/lib/python*/EXTERNALLY-MANAGED && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    # torch in the debian12 repo is too old for cellpose
    # pandas depends on sympy, and pip does not install torch if an outdated sympy is installed by apt
    pip3 install --no-cache-dir torch pandas numpy scipy numba git+https://github.com/olokelo/Pillow.git@jxl-support2 && \
    # dependencies for cellpose
    pip3 install --no-cache-dir tqdm opencv-python tifffile fastremap natsort roifile && \
    git clone https://github.com/MouseLand/cellpose.git && \
    mv cellpose/cellpose /usr/local/lib/python3.11/dist-packages/ && \
    rm -rf jxl-debs-amd64-debian-bookworm.tar.gz jxl-linux-x86_64-static.tar.gz tools LICENSE.* /var/lib/apt/lists/* *jxl*.deb cellpose

# the nvidia-container-toolkit allows using the GPU on Hail Batch container without installing the full driver
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt update  && \
    apt install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*
