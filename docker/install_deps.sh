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
  flex \
  bison \
  build-essential \
  git \
  wget \
  pciutils \
  xserver-xorg \
  xserver-xorg-video-fbdev \
  xauth \
  cmake \
  protobuf-compiler \
  libxml2-dev \
  libxslt-dev


# Eigen
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror
mkdir build
cd build
cmake ..
make -j8
sudo make install
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

# clone OMG-Planner
mkdir Projects
cd Projects
git clone https://github.com/liruiw/OMG-Planner.git --recursive
cd OMG-Planner

# Sophus
cd Sophus
mkdir build
cd build
cmake .. -Wno-error=deprecated-declarations -Wno-deprecated-declarations
make -j8
make install
cd ../..


# ycb render
cd ycb_render
python setup.py develop
cd ..

# omg layer
cd layers
python setup.py install
cd ..

# KDL
cd orocos_kinematics_dynamics
cd sip-4.19.3
python configure.py
make -j8; sudo make install

export ROS_PYTHON_VERSION=3
cd ../orocos_kdl
mkdir build; cd build;
cmake ..
make -j8; sudo make install

cd ../../python_orocos_kdl
mkdir build; cd build;
cmake ..  -DPYTHON_EXECUTABLE=~/usr/bin/python
make -j8;  cp PyKDL.so ~/anaconda2/envs/omg/lib/python3.6/site-packages/

# install PointNet
git clone https://github.com/liruiw/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install .

# install GA-DDPG  
cd ~/Projects
git clone https://github.com/liruiw/GA-DDPG.git  
cd  GA-DDPG
rm -rf OMG
ln -s ../OMG-Planner OMG

# install HCG  
cd ~/Projects
git clone https://github.com/liruiw/HCG.git  
cd  HCG
rm -rf OMG
ln -s ../OMG-Planner OMG