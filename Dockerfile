FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
RUN apt-get update

ENV CUDA_VERSION=cu118
ENV TORCH_VERSION=1.13.1
ENV ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt-get -y install python3.8
# RUN apt-get update && apt-get -y install python3-pip --fix-missing

RUN pip3 install torch==$TORCH_VERSION+$CUDA_VERSION torchvision==0.16.0+$CUDA_VERSION torchaudio==2.1.0+$CUDA_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION

RUN apt-get install -y \
    wget\
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# All users can use /home/user as their home directory.
RUN mkdir /home/user
ENV HOME=/home/user
RUN chmod 777 /home/user


# Install OpenCV3 Python bindings.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   libgtk2.0-0 \
   libcanberra-gtk-module \
&& sudo rm -rf /var/lib/apt/lists/*


RUN pip3 --no-cache-dir install pandas==2.1.2 scipy==1.11.3\
 && pip3 install --no-cache-dir pytorch-lightning==2.1.0

RUN pip3 --no-cache-dir install jupyter==1.0.0
RUN pip3 --no-cache-dir install matplotlib==3.8.0
RUN pip3 --no-cache-dir install pyod==1.1.1
RUN pip3 --no-cache-dir install numpy==1.23

# RUN pip install ray
# RUN pip install ray[tune]
# RUN pip install optuna

# Set the default command to python3.
CMD ["python3"]

#docker rm -f $(docker ps -a -q) && docker volume rm $(docker volume ls -q)