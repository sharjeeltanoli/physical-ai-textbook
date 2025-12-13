---
sidebar_position: 8
title: ROS 2 Best Practices & Debugging
description: Production-ready practices, debugging techniques, and common pitfalls
---

# ROS 2 بہترین طریقے اور ڈیبگنگ

## جائزہ

قابل اعتماد روبوٹ سسٹمز کی تعمیر کے لیے صرف فنکشنل کوڈ سے زیادہ کی ضرورت ہوتی ہے۔ یہ باب بہترین طریقوں، ڈیبگنگ کی حکمت عملیوں، ٹیسٹنگ کے طریقوں، اور عام غلطیوں سے بچنے کے طریقے بیان کرتا ہے۔

## کوڈ کی تنظیم کے بہترین طریقے

### پیکیج کی ساخت

my_robot_package/
├── config/                    # Configuration files
│   ├── params.yaml
│   └── rviz_config.rviz
├── launch/                    # Launch files
│   ├── robot.launch.py
│   └── simulation.launch.py
├── msg/                       # Custom messages
│   └── RobotStatus.msg
├── srv/                       # Custom services
│   └── SetMode.srv
├── action/                    # Custom actions
│   └── Navigate.action
├── include/my_robot_package/  # C++ headers
│   └── controller.hpp
├── src/                       # C++ source files
│   └── controller.cpp
├── my_robot_package/          # Python package
│   ├── __init__.py
│   ├── controller.py
│   └── utils.py
├── test/                      # Unit tests
│   ├── test_controller.py
│   └── test_kinematics.cpp
├── urdf/                      # Robot descriptions
│   └── robot.urdf.xacro
├── CMakeLists.txt             # C++ build config
├── package.xml                # Package manifest
├── setup.py                   # Python setup
└── README.md                  # Documentation

### نام رکھنے کے قواعد

#### پیکیجز
```
snake_case: my_robot_controller
```

#### نوڈز
```
snake_case: camera_driver_node
```

#### ٹاپکس
```
snake_case with namespaces: /robot/sensors/camera/image_raw
```

#### سروسز
```
snake_case with namespaces: /robot/set_velocity
```

#### میسج کی اقسام
```
CamelCase: RobotStatus, LaserScan
```

### نیم اسپیس کی تنظیم

```python
# Good: Organized namespaces
/robot1/sensors/lidar
/robot1/sensors/camera
/robot1/control/cmd_vel
/robot1/status

# Bad: Flat namespace
/lidar
/camera
/vel
/status
```

## لاگنگ کے بہترین طریقے

### لاگ لیولز

```python
class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # DEBUG: Detailed diagnostic information
        self.get_logger().debug('Entering loop iteration')

        # INFO: General informational messages
        self.get_logger().info('Node started successfully')

        # WARN: Warning messages (recoverable issues)
        self.get_logger().warn('Battery level low: 15%')

        # ERROR: Error messages (serious issues)
        self.get_logger().error('Failed to connect to hardware')

        # FATAL: Fatal errors (unrecoverable)
        self.get_logger().fatal('Critical hardware failure, shutting down')
```

### منظم لاگنگ

```python
# Good: Informative, structured logs
self.get_logger().info(
    f'Published velocity command: linear={twist.linear.x:.2f} m/s, '
    f'angular={twist.angular.z:.2f} rad/s'
)

# Bad: Vague logs
self.get_logger().info('Published')

# Good: Contextual error messages
self.get_logger().error(
    f'Failed to process image from camera {self.camera_id}: {str(e)}'
)

# Bad: Generic error
self.get_logger().error('Error occurred')
```

### تھروٹلڈ لاگنگ

```python
from rclpy.duration import Duration

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.last_log_time = self.get_clock().now()

    def high_frequency_callback(self):
        """Callback running at 100Hz"""

        # Only log once per second
        current_time = self.get_clock().now()
        if (current_time - self.last_log_time) > Duration(seconds=1.0):
            self.get_logger().info('Still processing...')
            self.last_log_time = current_time
```

## ایرر ہینڈلنگ

### گریسفل ڈیگریڈیشن

```python
class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        self.sensor_timeout = 5.0  # seconds
        self.last_sensor_time = self.get_clock().now()

        self.sensor_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.sensor_callback,
            10
        )

        self.timer = self.create_timer(1.0, self.check_health)

    def sensor_callback(self, msg):
        self.last_sensor_time = self.get_clock().now()
        # Process sensor data...

    def check_health(self):
        """Monitor system health"""
        time_since_sensor = (
            self.get_clock().now() - self.last_sensor_time
        ).nanoseconds / 1e9

        if time_since_sensor > self.sensor_timeout:
            self.get_logger().error(
                f'Sensor timeout: no data for {time_since_sensor:.1f}s'
            )
            # Enter safe mode
            self.enter_safe_mode()

    def enter_safe_mode(self):
        """Stop robot safely"""
        self.get_logger().warn('Entering safe mode')
        # Publish zero velocity, disable autonomous control, etc.
```

### سروس کال ایرر ہینڈلنگ

```python
def call_service_safely(self):
    """Robust service call with timeout and error handling"""

    # Wait for service with timeout
    if not self.client.wait_for_service(timeout_sec=5.0):
        self.get_logger().error('Service not available after 5s')
        return None

    # Create request
    request = MyService.Request()
    request.data = 42

    # Call service
    future = self.client.call_async(request)

    # Wait for response with timeout
    rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

    if future.done():
        try:
            response = future.result()
            return response
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None
    else:
        self.get_logger().error('Service call timed out')
        return None
```

## ٹیسٹنگ

### پائٹیسٹ کے ساتھ یونٹ ٹیسٹ

```python
# test/test_controller.py

import pytest
import rclpy
from my_robot_controller.controller import RobotController
from geometry_msgs.msg import Twist

@pytest.fixture
def node():
    """Create node for testing"""
    rclpy.init()
    node = RobotController()
    yield node
    node.destroy_node()
    rclpy.shutdown()

def test_velocity_clamping(node):
    """Test that velocities are clamped to safe limits"""

    # Create command exceeding limits
    cmd = Twist()
    cmd.linear.x = 10.0  # Exceeds max_linear_vel

    # Process command
    node.cmd_vel_callback(cmd)

    # Verify clamping
    assert node.current_linear_vel <= node.max_linear_vel

def test_emergency_stop(node):
    """Test emergency stop functionality"""

    # Activate emergency stop
    request = SetBool.Request()
    request.data = True
    response = SetBool.Response()

    node.emergency_stop_callback(request, response)

    # Verify stopped
    assert node.emergency_stop == True
    assert response.success == True

def test_odometry_calculation():
    """Test odometry calculation (pure function)"""

    from my_robot_controller.utils import calculate_odometry

    # Test case
    left_vel = 1.0
    right_vel = 1.0
    dt = 0.1
    wheel_base = 0.5

    x, y, theta = calculate_odometry(
        0.0, 0.0, 0.0,
        left_vel, right_vel,
        dt, wheel_base
    )

    # Verify straight line motion
    assert x == pytest.approx(0.1, rel=1e-3)
    assert y == pytest.approx(0.0, rel=1e-3)
    assert theta == pytest.approx(0.0, rel=1e-3)
```

### انٹیگریشن ٹیسٹ

```python
# test/test_integration.py

import unittest
import rclpy
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_testing.actions import ReadyToTest

class TestRobotSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_full_system(self):
        """Test complete robot system"""

        # Launch system
        # Publish commands
        # Verify expected behavior
        pass

def generate_test_description():
    return LaunchDescription([
        Node(
            package='my_robot_controller',
            executable='robot_controller',
            name='robot_controller'
        ),
        ReadyToTest()
    ])
```

### ٹیسٹ چلائیں

```bash
# Run all tests
colcon test --packages-select my_robot_package

# View test results
colcon test-result --all --verbose

# Run specific test
pytest test/test_controller.py::test_velocity_clamping
```

## ڈیبگنگ کی تکنیکیں

### کمانڈ لائن ڈیبگنگ

```bash
# List all topics with types
ros2 topic list -t

# Echo topic with timestamp
ros2 topic echo /odom --once

# Show topic bandwidth
ros2 topic bw /camera/image_raw

# Show topic publishing rate
ros2 topic hz /scan

# Show topic message definition
ros2 interface show sensor_msgs/msg/LaserScan

# Introspect node
ros2 node info /robot_controller

# Check parameter values
ros2 param list /robot_controller
ros2 param get /robot_controller max_speed

# Monitor service calls
ros2 service list
ros2 service type /emergency_stop

# Test service manually
ros2 service call /emergency_stop std_srvs/srv/SetBool "{data: true}"
```

### RQt ٹولز

```bash
# Node graph visualization
rqt_graph

# Topic monitoring
rqt_topic

# Service caller
rqt_service_caller

# Parameter reconfiguration
rqt_reconfigure

# Message publisher (manual testing)
rqt_publisher

# Console output
rqt_console

# Plot numeric data
rqt_plot /odom/pose/pose/position/x /odom/pose/pose/position/y
```

### پائتھون ڈیبگر انٹیگریشن

```python
# Insert breakpoint
import pdb; pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

```bash
# Run with debugger
python3 -m pdb $(ros2 pkg prefix my_package)/lib/my_package/my_node
```

### C++ کے لیے GDB

```bash
# Run with gdb
ros2 run --prefix 'gdb -ex run --args' my_package my_node

# Attach to running process
gdb -p $(pgrep my_node)
```

### میموری لیکس کے لیے Valgrind

```bash
# Check for memory leaks
ros2 run --prefix 'valgrind --leak-check=full' my_package my_node
```

## کارکردگی کی پروفائلنگ

### ٹاپک لیٹنسی

```bash
# Measure end-to-end latency
ros2 topic delay /camera/image_raw
```

### CPU پروفائلنگ (پائتھون)

```python
import cProfile
import pstats

def profile_node():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run node
    rclpy.spin(node)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### میموری پروفائلنگ

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    pass
```

## عام غلطیاں اور ان کے حل

### 1. QoS مس میچ

**مسئلہ:**
```python
# Publisher with RELIABLE
pub = self.create_publisher(LaserScan, 'scan', QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    depth=10
))

# Subscriber with BEST_EFFORT
sub = self.create_subscription(LaserScan, 'scan', callback, QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=10
))

# Result: No messages received!
```

**حل:**
```python
# Match QoS policies or use compatible ones
from rclpy.qos import qos_profile_sensor_data

pub = self.create_publisher(LaserScan, 'scan', qos_profile_sensor_data)
sub = self.create_subscription(LaserScan, 'scan', callback, qos_profile_sensor_data)
```

### 2. کال بیک بلاکنگ

**مسئلہ:**
```python
def callback(self, msg):
    # Long-running operation blocks executor!
    result = expensive_computation(msg)  # Takes 5 seconds
    self.process(result)
```

**حل:**
```python
import threading

def callback(self, msg):
    # Offload to separate thread
    thread = threading.Thread(target=self.process_async, args=(msg,))
    thread.start()

def process_async(self, msg):
    result = expensive_computation(msg)
    self.process(result)
```

### 3. ٹرانسفارم لوک اپ ٹائمنگ

**مسئلہ:**
```python
# Fails if transform not available yet
transform = self.tf_buffer.lookup_transform(
    'map',
    'base_link',
    rclpy.time.Time()  # Latest available
)
```

**حل:**
```python
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

try:
    # Wait for transform with timeout
    transform = self.tf_buffer.lookup_transform(
        'map',
        'base_link',
        rclpy.time.Time(),
        timeout=rclpy.duration.Duration(seconds=1.0)
    )
except (LookupException, ConnectivityException, ExtrapolationException) as e:
    self.get_logger().warn(f'Transform lookup failed: {e}')
    return
```

### 4. نوڈ کلین اپ

**مسئلہ:**
```python
# Nodes not destroyed properly
def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
    # Node not destroyed!
```

**حل:**
```python
def main():
    rclpy.init()
    node = MyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### 5. سرکولر ڈیپینڈینسیز

**مسئلہ:**
```xml
<!-- package_a/package.xml depends on package_b -->
<depend>package_b</depend>

<!-- package_b/package.xml depends on package_a -->
<depend>package_a</depend>
```

**حل:**
انٹرفیس پیکیج بنائیں:
```
Create interface package:
  robot_interfaces (messages only)
       ↑         ↑
       │         │
  package_a  package_b
```

## پروڈکشن چیک لسٹ

### تعیناتی سے پہلے

- [ ] تمام نوڈز میں مناسب لاگنگ ہو۔
- [ ] تمام ناکامی کے طریقوں کے لیے ایرر ہینڈلنگ ہو۔
- [ ] یونٹ ٹیسٹ 80% سے زیادہ کوریج کے ساتھ موجود ہوں۔
- [ ] اہم راستوں کے لیے انٹیگریشن ٹیسٹ موجود ہوں۔
- [ ] لانچ فائلیں پیرامیٹرائزڈ ہوں۔
- [ ] دستاویزات مکمل ہوں (README، تبصرے)۔
- [ ] QoS پالیسیاں ڈیٹا کی اقسام کے لیے مناسب ہوں۔
- [ ] وسائل کی حدود کا تجربہ کیا گیا ہو (CPU، میموری، بینڈوتھ)۔
- [ ] گریسفل شٹ ڈاؤن نافذ کیا گیا ہو۔
- [ ] ریکوری میکانزم موجود ہوں۔

### پروڈکشن میں نگرانی

- [ ] لاگ ایگریگیشن کنفیگر کی گئی ہو۔
- [ ] صحت کی نگرانی کا ڈیش بورڈ ہو۔
- [ ] اہم ناکامیوں کے لیے الرٹنگ ہو۔
- [ ] کارکردگی کے میٹرکس ٹریک کیے جائیں۔
- [ ] کریش ہونے پر خودکار ری اسٹارٹ ہو۔
- [ ] تشخیصی ٹاپکس شائع کیے گئے ہوں۔

## دستاویزات کے بہترین طریقے

### README ٹیمپلیٹ

```markdown
# Robot Controller Package

## Overview
Brief description of package functionality.

## Dependencies
- ROS 2 Humble
- geometry_msgs
- sensor_msgs

## Installation
\`\`\`bash
cd ~/ros2_ws/src
git clone ...
cd ~/ros2_ws
colcon build
\`\`\`

## Usage
\`\`\`bash
ros2 launch my_package robot.launch.py
\`\`\`

## Parameters
- `max_speed` (float, default: 1.0): Maximum linear velocity
- `wheel_base` (float, default: 0.5): Distance between wheels

## Topics
### Subscribed
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands

### Published
- `/odom` (nav_msgs/Odometry): Robot odometry

## Services
- `/emergency_stop` (std_srvs/SetBool): Emergency stop control

## License
Apache License 2.0
```

### کوڈ کے تبصرے

```python
class RobotController(Node):
    """
    Differential drive robot controller.

    Converts Twist commands to wheel velocities using differential
    drive kinematics. Implements emergency stop functionality.

    Subscribes:
        /cmd_vel (Twist): Velocity commands

    Publishes:
        /wheel_velocities (WheelVelocities): Individual wheel speeds

    Services:
        /emergency_stop (SetBool): Emergency stop control

    Parameters:
        wheel_separation (float): Distance between wheels (m)
        max_linear_vel (float): Maximum forward velocity (m/s)
    """

    def cmd_vel_callback(self, msg: Twist):
        """
        Process velocity command and compute wheel velocities.

        Args:
            msg: Twist message with linear and angular velocities

        The differential drive kinematics equations are:
            v_left = v - omega * L/2
            v_right = v + omega * L/2

        where v is linear velocity, omega is angular velocity,
        and L is wheel separation.
        """
        pass
```

## اہم نکات

- نام رکھنے کے مستقل اصولوں اور پیکیج کی ساخت پر عمل کریں۔
- مناسب لاگ لیولز اور منظم لاگنگ استعمال کریں۔
- مضبوط ایرر ہینڈلنگ اور گریسفل ڈیگریڈیشن نافذ کریں۔
- جامع یونٹ اور انٹیگریشن ٹیسٹ لکھیں۔
- ڈیبگنگ ٹولز (CLI, RQt, GDB, Valgrind) میں مہارت حاصل کریں۔
- عام غلطیوں سے بچیں (QoS مس میچ، بلاکنگ کال بیکس وغیرہ)۔
- کوڈ کو مکمل طور پر دستاویز کریں۔
- تعیناتی سے پہلے پروڈکشن چیک لسٹ استعمال کریں۔

## اگلا ماڈیول

مبارک ہو! آپ نے ROS 2 پر ماڈیول 1 مکمل کر لیا ہے۔ اگلا، ہم Gazebo اور Unity کا استعمال کرتے ہوئے روبوٹ سمولیشن کو دریافت کریں گے!

---

**پچھلا:** [Advanced ROS 2 Concepts](./advanced-ros2)
**اگلا:** [Module 2: Introduction to Robot Simulation](../module2/intro-simulation)
