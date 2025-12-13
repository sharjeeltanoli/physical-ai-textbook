---
sidebar_position: 3
title: Development Environment Setup
description: Step-by-step guide to setting up your robotics development environment
---

# ڈیویلپمنٹ ماحول کا سیٹ اپ

یہ گائیڈ آپ کو پورے کورس کے لیے اپنا ڈیویلپمنٹ ماحول کنفیگر کرنے میں مدد کرے گی۔ ہم تمام ضروری ٹولز، فریم ورک، اور ڈیپینڈنسیز کو سیٹ اپ کریں گے۔

## سسٹم کے تقاضے

### کم از کم تقاضے
- **OS**: Ubuntu 22.04 LTS (native یا WSL2)
- **CPU**: 4+ کور، 2.5 GHz+
- **RAM**: 16 GB
- **اسٹوریج**: 50 GB خالی جگہ
- **GPU**: اختیاری لیکن تجویز کردہ (CUDA سپورٹ کے ساتھ NVIDIA)

### تجویز کردہ خصوصیات
- **OS**: Ubuntu 22.04 LTS (native)
- **CPU**: 8+ کور، 3.0 GHz+
- **RAM**: 32 GB
- **اسٹوریج**: 100 GB SSD
- **GPU**: NVIDIA RTX 3060 یا بہتر (ماڈیول 3-4 کے لیے درکار ہے)

## آپریٹنگ سسٹم کا سیٹ اپ

### آپشن 1: اوبنٹو نیٹو (تجویز کردہ)

اگر آپ اوبنٹو 22.04 کو نیٹو طور پر چلا رہے ہیں، تو اگلے سیکشن پر جائیں۔

### آپشن 2: ونڈوز پر WSL2

bash
# Enable WSL2 (PowerShell as Administrator)
wsl --install -d Ubuntu-22.04

# Launch Ubuntu
wsl

# Update system
sudo apt update && sudo apt upgrade -y

### آپشن 3: ورچوئل مشین

- اوبنٹو 22.04 ISO ڈاؤن لوڈ کریں
- VirtualBox یا VMware استعمال کریں
- کم از کم 4 CPU کور، 16 GB RAM مختص کریں
- اگر دستیاب ہو تو 3D ایکسیلریشن کو فعال کریں

## لازمی ٹولز کی تنصیب

### 1. سسٹم پیکیجز کو اپ ڈیٹ کریں

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. بلڈ ایسینشلز انسٹال کریں

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

### 3. پائیتھون ڈیویلپمنٹ ٹولز انسٹال کریں

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

## ROS 2 کی تنصیب (ماڈیول 1)

### ROS 2 Humble انسٹال کریں

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

### ROS 2 ماحول کو کنفیگر کریں

```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify installation
ros2 --version
```

## سیمولیشن ٹولز (ماڈیول 2)

### Gazebo انسٹال کریں

```bash
# Install Gazebo Fortress
sudo apt install -y ros-humble-gazebo-ros-pkgs

# Verify installation
gazebo --version
```

### یونٹی ہب انسٹال کریں (اختیاری)

```bash
# Download Unity Hub
wget -O unity-hub.AppImage https://public-cdn.cloud.unity3d.com/hub/prod/UnityHub.AppImage

# Make executable
chmod +x unity-hub.AppImage

# Run Unity Hub
./unity-hub.AppImage
```

## NVIDIA ٹولز (ماڈیول 3)

### NVIDIA ڈرائیورز انسٹال کریں

```bash
# Check if NVIDIA GPU is available
lspci | grep -i nvidia

# Install drivers (if NVIDIA GPU detected)
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot
```

### CUDA ٹول کٹ انسٹال کریں

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

### Isaac Sim انسٹال کریں (ماڈیول 3 میں شامل کیا جائے گا)

ہدایات ماڈیول 3 میں فراہم کی جائیں گی، کیونکہ Isaac Sim کو اضافی سیٹ اپ کی ضرورت ہے۔

## پائیتھون لائبریریاں

### کور ML/AI لائبریریاں انسٹال کریں

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

### روبوٹکس کی مخصوص لائبریریاں انسٹال کریں

```bash
pip install \
  pyrealsense2 \
  open3d \
  pyrobot \
  modern-robotics
```

## ورژن کنٹرول اور تعاون

### Git کو کنفیگر کریں

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

## ڈیویلپمنٹ ٹولز

### VS Code انسٹال کریں (تجویز کردہ IDE)

```bash
# Download and install VS Code
sudo snap install --classic code

# Install useful extensions
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension ms-iot.vscode-ros
```

### متبادل: PyCharm

```bash
# Install PyCharm Community Edition
sudo snap install pycharm-community --classic
```

## ورک اسپیس کا سیٹ اپ

### ROS 2 ورک اسپیس بنائیں

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build

# Source workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

### کورس ریپوزیٹری کو کلون کریں

```bash
# Navigate to workspace src
cd ~/ros2_ws/src

# Clone course materials (placeholder URL)
# git clone https://github.com/your-org/physical-ai-course.git

# Build workspace with new package
cd ~/ros2_ws
colcon build
```

## تصدیق

### اپنی تنصیب کی جانچ کریں

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

## خرابیوں کا ازالہ

### عام مسائل

#### تنصیب کے بعد ROS 2 نہیں ملا
```bash
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

#### لانچ پر Gazebo کریش ہو جاتا ہے
```bash
# Disable GPU acceleration
export LIBGL_ALWAYS_SOFTWARE=1
gazebo
```

#### CUDA کا پتہ نہیں چلا
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall CUDA toolkit if needed
sudo apt install --reinstall cuda-toolkit-12-2
```

## اگلے اقدامات

مبارک ہو! آپ کا ڈیویلپمنٹ ماحول اب تیار ہے۔ اگلے ماڈیول میں، ہم ROS 2 میں گہرائی سے جائیں گے اور اپنی پہلی روبوٹ ایپلی کیشن بنائیں گے۔

## اضافی وسائل

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Gazebo Tutorials](https://gazebosim.org/docs)
- [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)
- [NVIDIA Developer Resources](https://developer.nvidia.com/)

---

**پچھلا:** [کورس کا جائزہ](./course-overview)
**اگلا:** [ماڈیول 1: ROS 2 کا تعارف](../module1/intro-ros2)
