---
sidebar_position: 16
title: Isaac ROS Integration
description: GPU-accelerated ROS 2 packages for robot perception
---

# Isaac ROS Integration

## Overview

Isaac ROS provides GPU-accelerated ROS 2 packages for perception, bringing NVIDIA AI capabilities to robotics applications.

## Isaac ROS GEMs

### Key Packages

**Perception:**
- `isaac_ros_visual_slam`: GPU-accelerated visual SLAM
- `isaac_ros_image_segmentation`: Semantic segmentation
- `isaac_ros_object_detection`: Object detection
- `isaac_ros_pose_estimation`: 6D pose estimation

**Acceleration:**
- `isaac_ros_dnn_inference`: DNN inference on Jetson/GPU
- `isaac_ros_image_proc`: Image processing
- `isaac_ros_compression`: Image compression

## Installation

### Prerequisites

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Install Isaac ROS

```bash
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
cd ~/ros2_ws
colcon build
```

## Visual SLAM Example

### Launch Visual SLAM

```bash
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
```

### Python Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

class SLAMUser(Node):
    def __init__(self):
        super().__init__('slam_user')

        # Subscribe to Visual SLAM output
        self.odom_sub = self.create_subscription(
            Odometry,
            '/visual_slam/tracking/odometry',
            self.odom_callback,
            10
        )

        # Publish camera images
        self.img_pub = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

    def odom_callback(self, msg):
        self.get_logger().info(
            f'Position: x={msg.pose.pose.position.x:.2f}, '
            f'y={msg.pose.pose.position.y:.2f}'
        )
```

## Object Detection

### Using DNN Inference

```bash
# Launch detection node
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py model_name:='peoplenet' model_repository_paths:=['/path/to/models']
```

### Custom Model

```python
from isaac_ros_dnn_inference import TensorRTInference

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # Initialize TensorRT inference
        self.inference = TensorRTInference(
            model_path='/path/to/model.plan',
            input_binding_names=['input'],
            output_binding_names=['output']
        )

    def process_image(self, image):
        # Run inference
        detections = self.inference.infer(image)
        return detections
```

## Semantic Segmentation

```python
from isaac_ros_image_segmentation import UNetSegmentation

class Segmenter(Node):
    def __init__(self):
        super().__init__('segmenter')

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.seg_pub = self.create_publisher(
            Image,
            '/segmentation/output',
            10
        )

        # Load model
        self.model = UNetSegmentation(model_path='/path/to/unet.plan')

    def image_callback(self, msg):
        # Perform segmentation
        segmentation_mask = self.model.segment(msg)
        self.seg_pub.publish(segmentation_mask)
```

## Integration with Isaac Sim

### Sim-to-ROS Bridge

```python
# In Isaac Sim
from omni.isaac.ros2_bridge import ROS2Bridge

# Create ROS 2 camera publisher
camera = Camera(prim_path="/World/Camera")
ros_camera = ROS2Bridge.create_camera_publisher(
    camera=camera,
    topic_name="/camera/image_raw"
)
```

## Performance Benchmarks

| Task | CPU (Hz) | GPU (Hz) | Speedup |
|------|----------|----------|---------|
| Visual SLAM | 10-15 | 60-90 | 6x |
| Object Detection | 5-10 | 30-60 | 6x |
| Segmentation | 2-5 | 30-45 | 10x |

## Key Takeaways

- Isaac ROS provides GPU-accelerated perception
- Seamless ROS 2 integration
- Pre-trained models available
- Significant performance improvements over CPU
- Works with Isaac Sim and real robots

---

**Previous:** [Isaac Gym for RL](./isaac-gym)
**Next:** [Advanced Isaac Features](./advanced-isaac)
