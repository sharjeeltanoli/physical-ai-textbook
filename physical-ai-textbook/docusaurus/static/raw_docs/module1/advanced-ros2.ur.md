---
sidebar_position: 7
title: Advanced ROS 2 Concepts
description: Actions, lifecycle nodes, component composition, and advanced patterns
---

# ROS 2 کے جدید تصورات

## جائزہ

یہ باب ROS 2 کی جدید خصوصیات کا احاطہ کرتا ہے جو پیداوار کے لیے تیار، قابل توسیع روبوٹ سسٹمز بنانے کے قابل بناتی ہیں۔ ہم Actions، Lifecycle Management، Component Composition، اور جدید کمیونیکیشن پیٹرنز کو تلاش کریں گے۔

## ROS 2 Actions

### Actions کیا ہیں؟

Actions **طویل مدتی کاموں** کے لیے ہیں جن کو ضرورت ہوتی ہے:
- **فیڈ بیک (Feedback)**: عمل کے دوران پیش رفت کی اپ ڈیٹس
- **نتیجہ (Result)**: مکمل ہونے پر حتمی نتیجہ
- **تنسیخ (Cancellation)**: عمل کے درمیان اسے روکنے کی صلاحیت

### Action بمقابلہ Topic بمقابلہ Service

| خصوصیت | Topic | Service | Action |
|---------|-------|---------|--------|
| **پیٹرن** | Pub-Sub | Request-Response | مقصد پر مبنی |
| **دورانیہ** | مسلسل | فوری | طویل مدتی |
| **فیڈ بیک** | نہیں | نہیں | **ہاں** |
| **قابل تنسیخ** | نہیں | نہیں | **ہاں** |
| **استعمال کا کیس** | سینسر ڈیٹا | کنفیگریشن | نیویگیشن، پکڑنا |

### Action کے اجزاء

```
Action Client                    Action Server
     │                                │
     ├─────── Goal ───────────────────>│
     │                                │ (Processing...)
     │<────── Feedback ────────────────│
     │<────── Feedback ────────────────│
     │                                │
     │───── Cancel (optional) ─────────>│
     │                                │
     │<────── Result ───────────────────│
```

### کسٹم Action کی تعریف کریں

```bash
# my_robot_interfaces/action/Navigate.action

# Goal
float32 target_x
float32 target_y
---
# Result
bool success
float32 final_x
float32 final_y
float32 distance_traveled
---
# Feedback
float32 current_x
float32 current_y
float32 distance_to_goal
float32 estimated_time_remaining
```

### Action Server کا نفاذ

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from my_robot_interfaces.action import Navigate
import math
import time

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')

        self._action_server = ActionServer(
            self,
            Navigate,
            'navigate_to_goal',
            self.execute_callback
        )

        self.get_logger().info('Navigation Action Server started')

    def execute_callback(self, goal_handle):
        """Execute navigation goal"""
        self.get_logger().info('Executing goal...')

        # Get goal
        target_x = goal_handle.request.target_x
        target_y = goal_handle.request.target_y

        # Simulated current position
        current_x, current_y = 0.0, 0.0

        feedback_msg = Navigate.Feedback()
        result = Navigate.Result()

        # Simulate navigation with feedback
        total_distance = math.sqrt(target_x**2 + target_y**2)
        steps = 10

        for i in range(steps):
            # Check if cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result.success = False
                return result

            # Simulate progress
            progress = (i + 1) / steps
            current_x = target_x * progress
            current_y = target_y * progress

            distance_to_goal = math.sqrt(
                (target_x - current_x)**2 + (target_y - current_y)**2
            )

            # Publish feedback
            feedback_msg.current_x = current_x
            feedback_msg.current_y = current_y
            feedback_msg.distance_to_goal = distance_to_goal
            feedback_msg.estimated_time_remaining = (steps - i) * 0.5

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Progress: {progress*100:.0f}%')

            time.sleep(0.5)

        # Goal succeeded
        goal_handle.succeed()

        result.success = True
        result.final_x = current_x
        result.final_y = current_y
        result.distance_traveled = total_distance

        self.get_logger().info('Goal succeeded!')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = NavigationActionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Action Client کا نفاذ

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_interfaces.action import Navigate

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')

        self._action_client = ActionClient(
            self,
            Navigate,
            'navigate_to_goal'
        )

    def send_goal(self, x, y):
        """Send navigation goal"""
        goal_msg = Navigate.Goal()
        goal_msg.target_x = x
        goal_msg.target_y = y

        self.get_logger().info(f'Sending goal: ({x}, {y})')

        # Wait for server
        self._action_client.wait_for_server()

        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance"""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: pos=({feedback.current_x:.2f}, {feedback.current_y:.2f}), '
            f'distance={feedback.distance_to_goal:.2f}, '
            f'ETA={feedback.estimated_time_remaining:.1f}s'
        )

    def get_result_callback(self, future):
        """Handle result"""
        result = future.result().result
        status = future.result().status

        if status == 4:  # SUCCEEDED
            self.get_logger().info(
                f'Goal succeeded! Final position: ({result.final_x:.2f}, {result.final_y:.2f}), '
                f'Distance traveled: {result.distance_traveled:.2f}m'
            )
        else:
            self.get_logger().error(f'Goal failed with status: {status}')

def main(args=None):
    rclpy.init(args=args)
    node = NavigationActionClient()

    # Send goal
    node.send_goal(10.0, 5.0)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Actions کو منسوخ کرنا

```python
# In action client
def cancel_goal(self):
    """Cancel current goal"""
    if self._goal_handle is not None:
        self.get_logger().info('Cancelling goal')
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

def cancel_done_callback(self, future):
    cancel_response = future.result()
    if len(cancel_response.goals_canceling) > 0:
        self.get_logger().info('Goal successfully canceled')
    else:
        self.get_logger().error('Goal failed to cancel')
```

## Lifecycle Nodes

### Lifecycle Nodes کیا ہیں؟

Lifecycle nodes منظم اسٹیٹ ٹرانزیشن کے ذریعے **مخصوص سٹارٹ اپ اور شٹ ڈاؤن** فراہم کرتے ہیں۔ یہ حفاظت اور بھروسے کے لیے اہم ہیں۔

### Lifecycle States

```
┌─────────────────┐
│   Unconfigured  │ (Initial state)
└─────────────────┘
        │ configure()
        ↓
┌─────────────────┐
│    Inactive     │ (Configured but not active)
└─────────────────┘
        │ activate()
        ↓
┌─────────────────┐
│     Active      │ (Running normally)
└─────────────────┘
        │ deactivate()
        ↓
┌─────────────────┐
│    Inactive     │
└─────────────────┘
        │ cleanup()
        ↓
┌─────────────────┐
│   Unconfigured  │
└─────────────────┘
        │ shutdown()
        ↓
┌─────────────────┐
│   Finalized     │ (Terminal state)
└─────────────────┘
```

### Lifecycle Node کا نفاذ

```python
import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from std_msgs.msg import String

class LifecyclePublisher(Node):
    def __init__(self):
        super().__init__('lifecycle_publisher')
        self.publisher_ = None
        self.timer = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure resources"""
        self.get_logger().info('Configuring...')

        # Create publisher
        self.publisher_ = self.create_lifecycle_publisher(
            String,
            'lifecycle_chatter',
            10
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate and start publishing"""
        self.get_logger().info('Activating...')

        # Start timer
        self.timer = self.create_timer(1.0, self.publish_callback)

        # Activate publisher
        self.publisher_.on_activate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate and stop publishing"""
        self.get_logger().info('Deactivating...')

        # Stop timer
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        # Deactivate publisher
        self.publisher_.on_deactivate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Clean up resources"""
        self.get_logger().info('Cleaning up...')

        # Destroy publisher
        self.destroy_lifecycle_publisher(self.publisher_)
        self.publisher_ = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown"""
        self.get_logger().info('Shutting down...')

        # Clean up if not already done
        if self.timer is not None:
            self.timer.cancel()

        if self.publisher_ is not None:
            self.destroy_lifecycle_publisher(self.publisher_)

        return TransitionCallbackReturn.SUCCESS

    def publish_callback(self):
        """Publish message"""
        msg = String()
        msg.data = f'Lifecycle message at {self.get_clock().now().to_msg()}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = LifecyclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Lifecycle کا انتظام کرنا

```bash
# Trigger state transitions
ros2 lifecycle set /lifecycle_publisher configure
ros2 lifecycle set /lifecycle_publisher activate

# Check current state
ros2 lifecycle get /lifecycle_publisher

# List available transitions
ros2 lifecycle list /lifecycle_publisher
```

## Component Composition

### Component Composition کیا ہے؟

ایک **سنگل پروسیس** میں متعدد nodes کو لوڈ کریں تاکہ:
- اوور ہیڈ (overhead) کم کریں (کوئی انٹر-پروسیس کمیونیکیشن نہیں)
- کارکردگی بہتر بنائیں
- تعیناتی (deployment) کو آسان بنائیں

### Component Node کا نفاذ

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherComponent(Node):
    """Component that can be loaded dynamically"""

    def __init__(self, options):
        super().__init__('publisher_component', options=options)

        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(1.0, self.publish_callback)

    def publish_callback(self):
        msg = String()
        msg.data = 'Component message'
        self.publisher_.publish(msg)

class SubscriberComponent(Node):
    """Component that can be loaded dynamically"""

    def __init__(self, options):
        super().__init__('subscriber_component', options=options)

        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

### Composition Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='my_package',
                plugin='PublisherComponent',
                name='publisher'
            ),
            ComposableNode(
                package='my_package',
                plugin='SubscriberComponent',
                name='subscriber'
            ),
        ],
        output='screen',
    )

    return LaunchDescription([container])
```

## Parameters اور Dynamic Reconfigure

### Parameters کا اعلان اور استعمال کریں

```python
class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with defaults
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('robot_name', 'robot1')
        self.declare_parameter('enable_safety', True)

        # Get parameter values
        self.max_speed = self.get_parameter('max_speed').value
        self.robot_name = self.get_parameter('robot_name').value
        self.enable_safety = self.get_parameter('enable_safety').value

        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(
            f'Parameters: max_speed={self.max_speed}, '
            f'robot_name={self.robot_name}, '
            f'enable_safety={self.enable_safety}'
        )

    def parameter_callback(self, params):
        """Handle parameter changes at runtime"""
        for param in params:
            if param.name == 'max_speed':
                if param.value <= 0.0:
                    # Reject invalid value
                    return SetParametersResult(successful=False)

                self.max_speed = param.value
                self.get_logger().info(f'Updated max_speed to {self.max_speed}')

            elif param.name == 'enable_safety':
                self.enable_safety = param.value
                self.get_logger().info(f'Safety: {self.enable_safety}')

        return SetParametersResult(successful=True)
```

### Parameter فائل (YAML)

```yaml
# config/robot_params.yaml

/**:
  ros__parameters:
    max_speed: 2.5
    robot_name: "speedy_robot"
    enable_safety: true

parameterized_node:
  ros__parameters:
    max_speed: 1.5
    robot_name: "careful_robot"
```

### Parameters لوڈ کریں

```bash
# Launch with parameter file
ros2 run my_package parameterized_node --ros-args --params-file config/robot_params.yaml

# Set parameter at runtime
ros2 param set /parameterized_node max_speed 3.0

# Get parameter
ros2 param get /parameterized_node max_speed

# List all parameters
ros2 param list
```

## tf2: Transform لائبریری

### tf2 کیا ہے؟

tf2 وقت کے ساتھ **کوارڈینیٹ فریم ٹرانسفارمیشنز** کو ٹریک کرتا ہے۔ یہ اس کے لیے ضروری ہے:
- سینسر فیوژن (Sensor Fusion)
- نیویگیشن (Navigation)
- مینیپولیشن (Manipulation)

### Transform براڈکاسٹ کریں

```python
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class TFBroadcasterNode(Node):
    def __init__(self):
        super().__init__('tf_broadcaster')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_timer_callback)

    def broadcast_timer_callback(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'robot'

        t.transform.translation.x = 1.0
        t.transform.translation.y = 0.5
        t.transform.translation.z = 0.0

        # Rotation (quaternion)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

### Transform سنیں

```python
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped

class TFListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.lookup_transform)

    def lookup_transform(self):
        try:
            # Lookup transform from 'world' to 'robot'
            transform = self.tf_buffer.lookup_transform(
                'world',
                'robot',
                rclpy.time.Time()
            )

            self.get_logger().info(
                f'Transform: x={transform.transform.translation.x:.2f}, '
                f'y={transform.transform.translation.y:.2f}'
            )

        except Exception as e:
            self.get_logger().warn(f'Could not transform: {e}')
```

## Multi-Threading اور Executors

### Single-Threaded Executor (ڈیفالٹ)

```python
import rclpy
from rclpy.executors import SingleThreadedExecutor

def main():
    rclpy.init()

    node1 = MyNode1()
    node2 = MyNode2()

    executor = SingleThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node1.destroy_node()
        node2.destroy_node()
        rclpy.shutdown()
```

### Multi-Threaded Executor

```python
from rclpy.executors import MultiThreadedExecutor

def main():
    rclpy.init()

    node = HighFrequencyNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
```

## بہترین طریقوں کا خلاصہ

✅ **طویل مدتی کاموں کے لیے actions کا استعمال کریں** جو فیڈ بیک اور تنسیخ کے ساتھ ہوں
✅ اہم سسٹم کے اجزاء کے لیے **lifecycle nodes کا نفاذ کریں**
✅ کارکردگی اہم ہونے پر **nodes کو کمپوز کریں**
✅ ہارڈ کوڈنگ کے بجائے کنفیگریشن کو **parameterize کریں**
✅ تمام کوارڈینیٹ ٹرانسفارمیشنز کے لیے **tf2 استعمال کریں**
✅ node کی ضروریات کی بنیاد پر **مناسب executors کا انتخاب کریں**

## اگلے اقدامات

اب آپ کے پاس ROS 2 کی جدید مہارتیں ہیں! اس ماڈیول کے آخری باب میں، ہم ڈیبگنگ، ٹیسٹنگ، اور پروڈکشن سسٹمز کے لیے بہترین طریقوں کا احاطہ کریں گے۔

---

**Previous:** [Building Your First Robot Application](./first-robot-app)
**Next:** [Best Practices & Debugging](./best-practices)
