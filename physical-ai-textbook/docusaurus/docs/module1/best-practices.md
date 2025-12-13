---
sidebar_position: 8
title: ROS 2 Best Practices & Debugging
description: Production-ready practices, debugging techniques, and common pitfalls
---

# ROS 2 Best Practices & Debugging

## Overview

Building reliable robot systems requires more than just functional code. This chapter covers best practices, debugging strategies, testing approaches, and common pitfalls to avoid.

## Code Organization Best Practices

### Package Structure

```
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
```

### Naming Conventions

#### Packages
```
snake_case: my_robot_controller
```

#### Nodes
```
snake_case: camera_driver_node
```

#### Topics
```
snake_case with namespaces: /robot/sensors/camera/image_raw
```

#### Services
```
snake_case with namespaces: /robot/set_velocity
```

#### Message Types
```
CamelCase: RobotStatus, LaserScan
```

### Namespace Organization

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

## Logging Best Practices

### Log Levels

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

### Structured Logging

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

### Throttled Logging

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

## Error Handling

### Graceful Degradation

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

### Service Call Error Handling

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

## Testing

### Unit Tests with pytest

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

### Integration Tests

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

### Run Tests

```bash
# Run all tests
colcon test --packages-select my_robot_package

# View test results
colcon test-result --all --verbose

# Run specific test
pytest test/test_controller.py::test_velocity_clamping
```

## Debugging Techniques

### Command-Line Debugging

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

### RQt Tools

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

### Python Debugger Integration

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

### GDB for C++

```bash
# Run with gdb
ros2 run --prefix 'gdb -ex run --args' my_package my_node

# Attach to running process
gdb -p $(pgrep my_node)
```

### Valgrind for Memory Leaks

```bash
# Check for memory leaks
ros2 run --prefix 'valgrind --leak-check=full' my_package my_node
```

## Performance Profiling

### Topic Latency

```bash
# Measure end-to-end latency
ros2 topic delay /camera/image_raw
```

### CPU Profiling (Python)

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

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    pass
```

## Common Pitfalls and Solutions

### 1. QoS Mismatch

**Problem:**
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

**Solution:**
```python
# Match QoS policies or use compatible ones
from rclpy.qos import qos_profile_sensor_data

pub = self.create_publisher(LaserScan, 'scan', qos_profile_sensor_data)
sub = self.create_subscription(LaserScan, 'scan', callback, qos_profile_sensor_data)
```

### 2. Callback Blocking

**Problem:**
```python
def callback(self, msg):
    # Long-running operation blocks executor!
    result = expensive_computation(msg)  # Takes 5 seconds
    self.process(result)
```

**Solution:**
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

### 3. Transform Lookup Timing

**Problem:**
```python
# Fails if transform not available yet
transform = self.tf_buffer.lookup_transform(
    'map',
    'base_link',
    rclpy.time.Time()  # Latest available
)
```

**Solution:**
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

### 4. Node Cleanup

**Problem:**
```python
# Nodes not destroyed properly
def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
    # Node not destroyed!
```

**Solution:**
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

### 5. Circular Dependencies

**Problem:**
```xml
<!-- package_a/package.xml depends on package_b -->
<depend>package_b</depend>

<!-- package_b/package.xml depends on package_a -->
<depend>package_a</depend>
```

**Solution:**
```
Create interface package:
  robot_interfaces (messages only)
       ↑         ↑
       │         │
  package_a  package_b
```

## Production Checklist

### Before Deployment

- [ ] All nodes have proper logging
- [ ] Error handling for all failure modes
- [ ] Unit tests with &gt;80% coverage
- [ ] Integration tests for critical paths
- [ ] Launch files parameterized
- [ ] Documentation complete (README, comments)
- [ ] QoS policies appropriate for data types
- [ ] Resource limits tested (CPU, memory, bandwidth)
- [ ] Graceful shutdown implemented
- [ ] Recovery mechanisms in place

### Monitoring in Production

- [ ] Log aggregation configured
- [ ] Health monitoring dashboard
- [ ] Alerting for critical failures
- [ ] Performance metrics tracked
- [ ] Automatic restarts on crashes
- [ ] Diagnostic topics published

## Documentation Best Practices

### README Template

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

### Code Comments

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

## Key Takeaways

- Follow consistent naming conventions and package structure
- Use appropriate log levels and structured logging
- Implement robust error handling and graceful degradation
- Write comprehensive unit and integration tests
- Master debugging tools (CLI, RQt, GDB, Valgrind)
- Avoid common pitfalls (QoS mismatch, blocking callbacks, etc.)
- Document code thoroughly
- Use production checklist before deployment

## Next Module

Congratulations! You've completed Module 1 on ROS 2. Next, we'll explore robot simulation using Gazebo and Unity!

---

**Previous:** [Advanced ROS 2 Concepts](./advanced-ros2)
**Next:** [Module 2: Introduction to Robot Simulation](../module2/intro-simulation)
