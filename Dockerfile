ARG TORCH_VERSION=1.13.1
# Always update SPCONV_VERSION equal to cuda version without .
ARG CUDA_VERSION=11.6
ARG CUDNN_VERSION=8

# ARG IMG_TAG=pointcept/pointcept:pytorch${BASE_TORCH_TAG}

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ENV SPCONV_VERSION=116


# Fix nvidia-key error issue (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/*.list

# Installing apt packages
RUN export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	git wget tmux vim zsh build-essential cmake ninja-build libopenblas-dev libsparsehash-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# Install Pointcept environment
RUN conda install h5py pyyaml -c anaconda -y
RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

RUN pip install --upgrade pip
RUN pip install torch-geometric

# ENV CUDA_VERSION=${CUDA_VERSION}
RUN pip install spconv-cu${SPCONV_VERSION}
RUN pip install open3d

# Build MinkowskiEngine
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
WORKDIR /workspace/MinkowskiEngine
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" python setup.py install --blas=openblas --force_cuda
WORKDIR /workspace

# Build pointops
RUN git clone https://github.com/Pointcept/Pointcept.git
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointops -v

# Build pointgroup_ops
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointgroup_ops -v

# Build swin3d
RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0" pip install -U git+https://github.com/microsoft/Swin3D.git -v

RUN git config --global --add safe.directory /workspaces/Pointcept && \
	git config --global --add safe.directory /workspace/Pointcept && \
	git config --global --add safe.directory /workspace

RUN pip install clearml wandb 