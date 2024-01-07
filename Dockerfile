FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Vienna
# ENV FORCE_CUDA="1"
# ENV MMCV_WITH_OPS=1
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install build-essential libgtk-3-dev
RUN apt update
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install jupyter
RUN pip install lightning tensorboard
RUN pip install pandas matplotlib
RUN pip install --upgrade "pyzmq<25" "jupyter_client<8"
RUN pip install plotly
RUN pip install opencv-python
RUN pip install torch_geometric
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install mmpose
RUN mim install mmcv
RUN apt-get -y install git wget
RUN mim install mmdet
RUN pip install tensorboard
RUN pip install wandb
RUN pip install einops
RUN pip install timm
RUN pip install tensorboard
ADD Makefile /workspace
# ADD requirements.txt /workspace
