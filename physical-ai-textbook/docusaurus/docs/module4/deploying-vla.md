---
sidebar_position: 21
title: Deploying VLA for Robotics
description: Practical deployment of VLA models to real robots
---

# Deploying VLA for Robotics

## Model Optimization

### Model Quantization

```python
import torch

# Post-training quantization
model_fp32 = VLAModel()
model_fp32.load_state_dict(torch.load('vla_model.pth'))
model_fp32.eval()

# Quantize to INT8
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'vla_model_int8.pth')

# Compare sizes
print(f"FP32 model size: {os.path.getsize('vla_model.pth') / 1e6:.2f} MB")
print(f"INT8 model size: {os.path.getsize('vla_model_int8.pth') / 1e6:.2f} MB")
```

### Model Pruning

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """Remove unimportant weights"""

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    return model

pruned_model = prune_model(model, amount=0.3)
```

### TensorRT Conversion

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, trt_path):
    """Convert ONNX model to TensorRT for NVIDIA GPUs"""

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Enable FP16 precision
    config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())
```

## ROS 2 Integration

### VLA ROS Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch

class VLAROSNode(Node):
    def __init__(self):
        super().__init__('vla_node')

        # Load VLA model
        self.model = VLAModel()
        self.model.load_state_dict(torch.load('vla_model.pth'))
        self.model.eval()
        self.model.cuda()

        # CV Bridge
        self.bridge = CvBridge()

        # Current state
        self.current_image = None
        self.current_instruction = "Wait for command"

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.instruction_sub = self.create_subscription(
            String,
            '/robot/instruction',
            self.instruction_callback,
            10
        )

        # Publishers
        self.action_pub = self.create_publisher(
            Twist,
            '/robot/cmd_vel',
            10
        )

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info('VLA Node started')

    def image_callback(self, msg):
        """Receive camera image"""
        self.current_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def instruction_callback(self, msg):
        """Receive language instruction"""
        self.current_instruction = msg.data
        self.get_logger().info(f'New instruction: {self.current_instruction}')

    def control_loop(self):
        """Main control loop"""
        if self.current_image is None:
            return

        # Prepare inputs
        image_tensor = self.preprocess_image(self.current_image)
        text_tokens = self.tokenize_instruction(self.current_instruction)

        # Predict action
        with torch.no_grad():
            action = self.model(image_tensor, text_tokens)

        # Convert to robot command
        cmd_vel = self.action_to_twist(action)

        # Publish
        self.action_pub.publish(cmd_vel)

    def preprocess_image(self, image):
        """Preprocess image for model"""
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0).cuda()

    def tokenize_instruction(self, instruction):
        """Tokenize text instruction"""
        # Use tokenizer (e.g., from transformers)
        tokens = tokenizer(instruction, return_tensors='pt', padding=True)
        return tokens.input_ids.cuda()

    def action_to_twist(self, action):
        """Convert model output to Twist message"""
        twist = Twist()
        twist.linear.x = float(action[0, 0])
        twist.angular.z = float(action[0, 1])
        return twist

def main(args=None):
    rclpy.init(args=args)
    node = VLAROSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Safety Mechanisms

### Action Filtering

```python
class SafetyFilter:
    def __init__(self, max_velocity=1.0, max_acceleration=0.5):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.prev_action = None

    def filter_action(self, action):
        """Ensure action is safe"""

        # Clamp velocity
        action = torch.clamp(action, -self.max_velocity, self.max_velocity)

        # Limit acceleration
        if self.prev_action is not None:
            delta = action - self.prev_action
            delta = torch.clamp(delta, -self.max_acceleration, self.max_acceleration)
            action = self.prev_action + delta

        self.prev_action = action
        return action
```

### Collision Avoidance

```python
class CollisionChecker:
    def __init__(self, robot):
        self.robot = robot

    def is_safe_action(self, action):
        """Check if action will cause collision"""

        # Simulate one step ahead
        predicted_state = self.robot.predict_next_state(action)

        # Check for collisions
        for obstacle in self.robot.get_obstacles():
            if self.robot.check_collision(predicted_state, obstacle):
                return False

        return True

    def get_safe_action(self, proposed_action):
        """Modify action to avoid collision"""

        if self.is_safe_action(proposed_action):
            return proposed_action

        # Find safe alternative (e.g., stop)
        return torch.zeros_like(proposed_action)
```

### Emergency Stop

```python
class EmergencyStop:
    def __init__(self):
        self.emergency_active = False

        # Subscribe to emergency stop button
        self.estop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.estop_callback,
            10
        )

    def estop_callback(self, msg):
        self.emergency_active = msg.data

    def check_and_stop(self, action):
        """Check emergency stop and override action"""
        if self.emergency_active:
            self.get_logger().warn('EMERGENCY STOP ACTIVE')
            return torch.zeros_like(action)
        return action
```

## Monitoring and Logging

```python
class VLAMonitor:
    def __init__(self):
        self.action_history = []
        self.success_count = 0
        self.failure_count = 0

    def log_action(self, instruction, image, action, success):
        """Log action for analysis"""

        self.action_history.append({
            'timestamp': time.time(),
            'instruction': instruction,
            'image': image,
            'action': action.cpu().numpy(),
            'success': success
        })

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_statistics(self):
        """Get performance statistics"""
        total = self.success_count + self.failure_count
        success_rate = self.success_count / total if total > 0 else 0

        return {
            'total_actions': total,
            'success_rate': success_rate,
            'recent_actions': self.action_history[-10:]
        }
```

## Edge Deployment

### NVIDIA Jetson

```python
# Optimize for Jetson
model = VLAModel()
model.half()  # FP16 for speed
model.eval()

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_image, dummy_text),
    "vla_model.onnx",
    opset_version=11
)

# Convert to TensorRT (Jetson-optimized)
trtexec --onnx=vla_model.onnx \
        --saveEngine=vla_model.trt \
        --fp16 \
        --workspace=4096
```

## Key Takeaways

- Optimize models via quantization, pruning, TensorRT
- Integrate with ROS 2 for robot control
- Implement safety filters and collision avoidance
- Add emergency stop mechanisms
- Monitor performance and log actions
- Deploy to edge devices (Jetson) for real-time performance

---

**Previous:** [Training VLA Models](./training-vla)
**Next:** [Real-World VLA Applications](./vla-applications)
