ARG TORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

# ARG IMG_TAG=pointcept/pointcept:pytorch${BASE_TORCH_TAG}

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel


# Fix nvidia-key error issue (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/*.list

# Installing apt packages
RUN export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	git wget build-essential cmake ninja-build libopenblas-dev libsparsehash-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# Installing apt packages
RUN conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# RUN conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda install h5py pyyaml -c anaconda -y
RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
RUN pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
RUN pip install spconv-cu117

# PPT (clip)
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
# docker & multi GPU arch
# RUN TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
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

# Open3D (visualization, optional)
RUN pip install open3d wandb clearml 
RUN pip install flash-attn --no-build-isolation