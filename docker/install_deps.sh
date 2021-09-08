#!/bin/bash

set -euxo pipefail
 
apt-get update
DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends \
  curl \
  terminator \
  tmux \
  vim \
  libsm6 \
  libxext6 \
  libxrender-dev \
  gedit \
  git \
  openssh-client \
  unzip \
  htop \
  libopenni-dev \
  apt-utils \
  usbutils \
  dialog \
  python3-virtualenv \
  python3-dev \
  python3-pip \
  ffmpeg \
  nvidia-settings \
  libffi-dev \
  build-essential \
  git \
  wget \
  pciutils \
  xserver-xorg \
  xserver-xorg-video-fbdev \
  xauth \
  protobuf-compiler \
  libxml2-dev \
  libxslt-dev \
  libglfw3-dev \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  freeglut3-dev


# clone OMG-Planner
mkdir Projects
cd Projects
git clone https://github.com/liruiw/OMG-Planner.git --recursive
cd OMG-Planner

# install python requirements
pip install --upgrade pip==21.2.4
pip install -U setuptools
pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0  -f https://download.pytorch.org/whl/torch_stable.html
 
# Eigen
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror
mkdir build
cd build
cmake ..
make -j8
sudo make install
make install
cd ../..

# Assimp
git clone https://github.com/assimp/assimp.git
cd assimp
git checkout v4.1.0
mkdir build
cd build
cmake ..
make -j8
make install
cd ../..

# Sophus
cd Sophus
mkdir build
cd build
cmake ..  
make -j8
make install
cd ../..

# ycb render
cd ycb_render
python3 setup.py develop
cd ..

# omg layer
cd layers
python3 setup.py install
cd ..

# KDL
cd orocos_kinematics_dynamics
cd sip-4.19.3
python3 configure.py
make -j8; sudo make install

export ROS_PYTHON_VERSION=3
cd ../orocos_kdl
mkdir build; cd build;
cmake ..
make -j8; sudo make install

cd ../../python_orocos_kdl
mkdir build; cd build;
cmake ..   
make -j8;   
make install

cd ../../..

bash download_data.sh
