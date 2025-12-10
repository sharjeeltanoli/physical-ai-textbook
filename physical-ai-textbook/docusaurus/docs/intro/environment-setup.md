---
sidebar_position: 3
title: Development Environment Setup
description: Step-by-step guide to setting up your robotics development environment
---

# Development Environment Setup

This guide will help you configure your development environment for the entire course. We'll set up all necessary tools, frameworks, and dependencies.

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 22.04 LTS (native or WSL2)
- **CPU**: 4+ cores, 2.5 GHz+
- **RAM**: 16 GB
- **Storage**: 50 GB free space
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

### Recommended Specifications
- **OS**: Ubuntu 22.04 LTS (native)
- **CPU**: 8+ cores, 3.0 GHz+
- **RAM**: 32 GB
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA RTX 3060 or better (required for Module 3-4)

## Operating System Setup

### Option 1: Ubuntu Native (Recommended)

If you're running Ubuntu 22.04 natively, skip to the next section.

### Option 2: WSL2 on Windows

```bash
# Enable WSL2 (PowerShell as Administrator)
wsl --install -d Ubuntu-22.04

# Launch Ubuntu
wsl

# Update system
sudo apt update && sudo apt upgrade -y
```

### Option 3: Virtual Machine

- Download Ubuntu 22.04 ISO
- Use VirtualBox or VMware
- Allocate at least 4 CPU cores, 16 GB RAM
- Enable 3D acceleration if available

## Essential Tools Installation

### 1. Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Build Essentials

```bash
sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  curl \
  vim \
  python3-pip \
  python3-venv
```

### 3. Install Python Development Tools

```bash
# Install Python 3.10 and development headers
sudo apt install -y python3.10 python3.10-dev python3.10-venv

# Create a virtual environment
mkdir -p ~/robotics-env
python3.10 -m venv ~/robotics-env

# Activate virtual environment
source ~/robotics-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## ROS 2 Installation (Module 1)

### Install ROS 2 Humble

```bash
# Set locale
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop

# Install development tools
sudo apt install -y \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep

# Initialize rosdep
sudo rosdep init
rosdep update
```

### Configure ROS 2 Environment

```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify installation
ros2 --version
```

## Simulation Tools (Module 2)

### Install Gazebo

```bash
# Install Gazebo Fortress
sudo apt install -y ros-humble-gazebo-ros-pkgs

# Verify installation
gazebo --version
```

### Install Unity Hub (Optional)

```bash
# Download Unity Hub
wget -O unity-hub.AppImage https://public-cdn.cloud.unity3d.com/hub/prod/UnityHub.AppImage

# Make executable
chmod +x unity-hub.AppImage

# Run Unity Hub
./unity-hub.AppImage
```

## NVIDIA Tools (Module 3)

### Install NVIDIA Drivers

```bash
# Check if NVIDIA GPU is available
lspci | grep -i nvidia

# Install drivers (if NVIDIA GPU detected)
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot
```

### Install CUDA Toolkit

```bash
# Install CUDA 12.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Install Isaac Sim (Will be covered in Module 3)

Instructions will be provided in Module 3, as Isaac Sim requires additional setup.

## Python Libraries

### Install Core ML/AI Libraries

```bash
# Activate virtual environment
source ~/robotics-env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install common libraries
pip install \
  numpy \
  matplotlib \
  opencv-python \
  scipy \
  pandas \
  transformers \
  gymnasium
```

### Install Robotics-Specific Libraries

```bash
pip install \
  pyrealsense2 \
  open3d \
  pyrobot \
  modern-robotics
```

## Version Control & Collaboration

### Configure Git

```bash
# Install Git (if not already installed)
sudo apt install -y git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Generate SSH key for GitHub
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add SSH key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Display public key (add this to GitHub)
cat ~/.ssh/id_ed25519.pub
```

## Development Tools

### Install VS Code (Recommended IDE)

```bash
# Download and install VS Code
sudo snap install --classic code

# Install useful extensions
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension ms-iot.vscode-ros
```

### Alternative: PyCharm

```bash
# Install PyCharm Community Edition
sudo snap install pycharm-community --classic
```

## Workspace Setup

### Create ROS 2 Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build

# Source workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

### Clone Course Repository

```bash
# Navigate to workspace src
cd ~/ros2_ws/src

# Clone course materials (placeholder URL)
# git clone https://github.com/your-org/physical-ai-course.git

# Build workspace with new package
cd ~/ros2_ws
colcon build
```

## Verification

### Test Your Installation

```bash
# Test ROS 2
ros2 run demo_nodes_cpp talker

# In another terminal
ros2 run demo_nodes_py listener

# Test Gazebo
gazebo

# Test Python environment
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

#### ROS 2 not found after installation
```bash
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

#### Gazebo crashes on launch
```bash
# Disable GPU acceleration
export LIBGL_ALWAYS_SOFTWARE=1
gazebo
```

#### CUDA not detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall CUDA toolkit if needed
sudo apt install --reinstall cuda-toolkit-12-2
```

## Next Steps

Congratulations! Your development environment is now ready. In the next module, we'll dive into ROS 2 and build our first robot application.

## Additional Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Gazebo Tutorials](https://gazebosim.org/docs)
- [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)
- [NVIDIA Developer Resources](https://developer.nvidia.com/)

---

**Previous:** [Course Overview](./course-overview)
**Next:** [Module 1: Introduction to ROS 2](../module1/intro-ros2)
