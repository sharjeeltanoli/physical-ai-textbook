---
sidebar_position: 16
title: Isaac ROS Integration
description: GPU-accelerated ROS 2 packages for robot perception
---

# Isaac ROS انضمام

## جائزہ

Isaac ROS بصری ادراک کے لیے GPU-accelerated ROS 2 پیکیجز فراہم کرتا ہے، جو NVIDIA AI کی صلاحیتوں کو روبوٹکس ایپلی کیشنز تک پہنچاتا ہے۔

## Isaac ROS GEMs

### اہم پیکیجز

**بصری ادراک:**
- `isaac_ros_visual_slam`: GPU-accelerated visual SLAM
- `isaac_ros_image_segmentation`: Semantic segmentation
- `isaac_ros_object_detection`: Object detection
- `isaac_ros_pose_estimation`: 6D pose estimation

**تیزی:**
- `isaac_ros_dnn_inference`: Jetson/GPU پر DNN inference
- `isaac_ros_image_proc`: Image processing
- `isaac_ros_compression`: Image compression

## تنصیب

### پیشگی ضروریات

bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

### Isaac ROS انسٹال کریں

```bash
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
cd ~/ros2_ws
colcon build
```

## Visual SLAM کی مثال

### Visual SLAM لانچ کریں

```bash
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
```

### پائتھون Node

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

### DNN Inference کا استعمال کرتے ہوئے

```bash
# Launch detection node
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py model_name:='peoplenet' model_repository_paths:=['/path/to/models']
```

### کسٹم ماڈل

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

## Isaac Sim کے ساتھ انضمام

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

## کارکردگی کے معیار

| ٹاسک | CPU (Hz) | GPU (Hz) | رفتار میں اضافہ |
|------|----------|----------|---------|
| Visual SLAM | 10-15 | 60-90 | 6x |
| Object Detection | 5-10 | 30-60 | 6x |
| Segmentation | 2-5 | 30-45 | 10x |

## اہم نکات

- Isaac ROS GPU-accelerated بصری ادراک فراہم کرتا ہے
- بغیر رکاوٹ ROS 2 انضمام
- پہلے سے تربیت یافتہ ماڈلز دستیاب ہیں
- CPU کے مقابلے میں کارکردگی میں نمایاں بہتری
- Isaac Sim اور حقیقی روبوٹس کے ساتھ کام کرتا ہے

---

**پچھلا:** [Isaac Gym for RL](./isaac-gym)
**اگلا:** [Advanced Isaac Features](./advanced-isaac)
