---
sidebar_position: 6
title: Building Your First Robot Application
description: Create a complete robot control system from scratch
---

# اپنی پہلی روبوٹ ایپلیکیشن بنانا

## پروجیکٹ کا جائزہ

اس باب میں، ہم ایک مکمل روبوٹ ایپلیکیشن بنائیں گے: ایک **differential drive robot controller**۔ یہ پروجیکٹ ایک فعال سسٹم بنانے کے لیے متعدد Nodes، Topics، اور Services کو مربوط کرے گا۔

### ہم کیا بنا رہے ہیں

User Input → Teleop Node → /cmd_vel → Robot Controller → Motor Drivers
                                ↓
                          Odometry Node → /odom → State Monitor

**Components:**
1.  **Teleoperation Node**: کی بورڈ ان پٹ حاصل کرتا ہے۔
2.  **Robot Controller Node**: ویلوسٹی کمانڈز پر عمل کرتا ہے۔
3.  **Odometry Node**: روبوٹ کی پوزیشن کو ٹریک کرتا ہے۔
4.  **State Monitor Node**: روبوٹ کا اسٹیٹس دکھاتا ہے۔
5.  **Emergency Stop Service**: حفاظتی میکانزم۔

## پروجیکٹ کا ڈھانچہ

### پیکیج بنائیں

```bash
cd ~/ros2_ws/src

# Create Python package
ros2 pkg create --build-type ament_python my_robot_controller \
  --dependencies rclpy geometry_msgs sensor_msgs std_msgs

# Create custom messages package
ros2 pkg create --build-type ament_cmake my_robot_interfaces

cd ~/ros2_ws
colcon build
source install/setup.bash
```

### پیکیج کا ڈھانچہ

```
my_robot_controller/
├── my_robot_controller/
│   ├── __init__.py
│   ├── teleop_node.py
│   ├── robot_controller.py
│   ├── odometry_node.py
│   └── state_monitor.py
├── launch/
│   └── robot_system.launch.py
├── config/
│   └── robot_params.yaml
├── package.xml
└── setup.py
```

## مرحلہ 1: کسٹم Messages کی وضاحت کریں

### روبوٹ اسٹیٹس Message

```bash
# my_robot_interfaces/msg/RobotStatus.msg

Header header
string robot_name
float32 battery_percentage
float32 x_position
float32 y_position
float32 theta_orientation
bool is_moving
uint8 error_code

# Error codes
uint8 ERROR_NONE=0
uint8 ERROR_LOW_BATTERY=1
uint8 ERROR_MOTOR_FAULT=2
uint8 ERROR_EMERGENCY_STOP=3
```

### پہیوں کی ویلوسٹیز Message

```bash
# my_robot_interfaces/msg/WheelVelocities.msg

float32 left_velocity
float32 right_velocity
```

### Messages بلڈ کریں

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_interfaces
source install/setup.bash

# Verify
ros2 interface show my_robot_interfaces/msg/RobotStatus
```

## مرحلہ 2: Teleoperation Node

### عمل درآمد

```python
# my_robot_controller/teleop_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Parameters
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 1.0)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        self.get_logger().info('Teleop Node Started')
        self.get_logger().info('Use WASD keys to control the robot')
        self.get_logger().info('Q to quit')

        # Start keyboard capture
        self.run()

    def get_key(self):
        """Capture a single keypress"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        """Main control loop"""
        try:
            while rclpy.ok():
                key = self.get_key()

                twist = Twist()

                if key == 'w':
                    twist.linear.x = self.linear_speed
                    self.get_logger().info('Forward')
                elif key == 's':
                    twist.linear.x = -self.linear_speed
                    self.get_logger().info('Backward')
                elif key == 'a':
                    twist.angular.z = self.angular_speed
                    self.get_logger().info('Left')
                elif key == 'd':
                    twist.angular.z = -self.angular_speed
                    self.get_logger().info('Right')
                elif key == ' ':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info('Stop')
                elif key == 'q':
                    self.get_logger().info('Quitting...')
                    break
                else:
                    continue

                self.cmd_vel_pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## مرحلہ 3: Robot Controller Node

### عمل درآمد

```python
# my_robot_controller/robot_controller.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from my_robot_interfaces.msg import WheelVelocities
from std_srvs.srv import SetBool
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Parameters
        self.declare_parameter('wheel_separation', 0.5)  # meters
        self.declare_parameter('wheel_radius', 0.1)      # meters
        self.declare_parameter('max_linear_vel', 2.0)    # m/s
        self.declare_parameter('max_angular_vel', 3.0)   # rad/s

        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value

        # State
        self.emergency_stop = False

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publisher for wheel velocities
        self.wheel_vel_pub = self.create_publisher(
            WheelVelocities,
            'wheel_velocities',
            10
        )

        # Emergency stop service
        self.emergency_stop_service = self.create_service(
            SetBool,
            'emergency_stop',
            self.emergency_stop_callback
        )

        self.get_logger().info('Robot Controller Started')

    def cmd_vel_callback(self, msg: Twist):
        """Convert Twist to wheel velocities"""

        if self.emergency_stop:
            self.get_logger().warn('Emergency stop active, ignoring command')
            return

        # Clamp velocities to safe limits
        linear = max(min(msg.linear.x, self.max_linear_vel), -self.max_linear_vel)
        angular = max(min(msg.angular.z, self.max_angular_vel), -self.max_angular_vel)

        # Differential drive kinematics
        # v_left = (linear - angular * wheel_separation / 2) / wheel_radius
        # v_right = (linear + angular * wheel_separation / 2) / wheel_radius

        v_left = linear - (angular * self.wheel_separation / 2.0)
        v_right = linear + (angular * self.wheel_separation / 2.0)

        # Publish wheel velocities
        wheel_msg = WheelVelocities()
        wheel_msg.left_velocity = v_left
        wheel_msg.right_velocity = v_right

        self.wheel_vel_pub.publish(wheel_msg)

        self.get_logger().debug(
            f'Cmd: linear={linear:.2f}, angular={angular:.2f} -> '
            f'Wheels: left={v_left:.2f}, right={v_right:.2f}'
        )

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop service"""
        self.emergency_stop = request.data

        if self.emergency_stop:
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            # Send zero velocity
            wheel_msg = WheelVelocities()
            wheel_msg.left_velocity = 0.0
            wheel_msg.right_velocity = 0.0
            self.wheel_vel_pub.publish(wheel_msg)
            response.success = True
            response.message = "Emergency stop activated"
        else:
            self.get_logger().info('Emergency stop deactivated')
            response.success = True
            response.message = "Emergency stop deactivated"

        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## مرحلہ 4: Odometry Node

### عمل درآمد

```python
# my_robot_controller/odometry_node.py

import rclpy
from rclpy.node import Node
from my_robot_interfaces.msg import WheelVelocities
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster
import math

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Parameters
        self.declare_parameter('wheel_separation', 0.5)
        self.declare_parameter('update_rate', 50.0)  # Hz

        self.wheel_separation = self.get_parameter('wheel_separation').value
        update_rate = self.get_parameter('update_rate').value

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_time = self.get_clock().now()

        # Subscriber
        self.wheel_vel_sub = self.create_subscription(
            WheelVelocities,
            'wheel_velocities',
            self.wheel_vel_callback,
            10
        )

        # Publisher
        self.odom_pub = self.create_publisher(
            Odometry,
            'odom',
            10
        )

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing odometry
        self.timer = self.create_timer(1.0 / update_rate, self.publish_odometry)

        self.current_left_vel = 0.0
        self.current_right_vel = 0.0

        self.get_logger().info('Odometry Node Started')

    def wheel_vel_callback(self, msg: WheelVelocities):
        """Store current wheel velocities"""
        self.current_left_vel = msg.left_velocity
        self.current_right_vel = msg.right_velocity

    def publish_odometry(self):
        """Calculate and publish odometry"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Calculate linear and angular velocity
        v = (self.current_right_vel + self.current_left_vel) / 2.0
        omega = (self.current_right_vel - self.current_left_vel) / self.wheel_separation

        # Update pose
        delta_x = v * math.cos(self.theta) * dt
        delta_y = v * math.sin(self.theta) * dt
        delta_theta = omega * dt

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta

        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Create and publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = euler_to_quaternion(0, 0, self.theta)

        # Velocity
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = omega

        self.odom_pub.publish(odom_msg)

        # Broadcast TF transform
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = euler_to_quaternion(0, 0, self.theta)

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## مرحلہ 5: State Monitor Node

### عمل درآمد

```python
# my_robot_controller/state_monitor.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from my_robot_interfaces.msg import RobotStatus
from std_msgs.msg import Header

class StateMonitor(Node):
    def __init__(self):
        super().__init__('state_monitor')

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Publisher
        self.status_pub = self.create_publisher(
            RobotStatus,
            'robot_status',
            10
        )

        # Timer for status publishing
        self.timer = self.create_timer(1.0, self.publish_status)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_vel = 0.0

        self.get_logger().info('State Monitor Started')

    def odom_callback(self, msg: Odometry):
        """Update state from odometry"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

        self.linear_vel = msg.twist.twist.linear.x

    def publish_status(self):
        """Publish robot status"""
        status_msg = RobotStatus()
        status_msg.header = Header()
        status_msg.header.stamp = self.get_clock().now().to_msg()

        status_msg.robot_name = "DifferentialDriveRobot"
        status_msg.battery_percentage = 85.0  # Placeholder
        status_msg.x_position = self.x
        status_msg.y_position = self.y
        status_msg.theta_orientation = self.theta
        status_msg.is_moving = abs(self.linear_vel) > 0.01
        status_msg.error_code = RobotStatus.ERROR_NONE

        self.status_pub.publish(status_msg)

        self.get_logger().info(
            f'Status: pos=({self.x:.2f}, {self.y:.2f}), '
            f'theta={self.theta:.2f}, moving={status_msg.is_moving}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = StateMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## مرحلہ 6: Launch File

### Launch File بنائیں

```python
# launch/robot_system.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'wheel_separation',
            default_value='0.5',
            description='Distance between wheels (meters)'
        ),

        # Robot Controller Node
        Node(
            package='my_robot_controller',
            executable='robot_controller',
            name='robot_controller',
            parameters=[{
                'wheel_separation': LaunchConfiguration('wheel_separation'),
                'wheel_radius': 0.1,
                'max_linear_vel': 2.0,
                'max_angular_vel': 3.0
            }],
            output='screen'
        ),

        # Odometry Node
        Node(
            package='my_robot_controller',
            executable='odometry_node',
            name='odometry_node',
            parameters=[{
                'wheel_separation': LaunchConfiguration('wheel_separation'),
                'update_rate': 50.0
            }],
            output='screen'
        ),

        # State Monitor Node
        Node(
            package='my_robot_controller',
            executable='state_monitor',
            name='state_monitor',
            output='screen'
        ),
    ])
```

## مرحلہ 7: پیکیج کی ترتیب (Package Configuration)

### setup.py کو اپ ڈیٹ کریں

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Differential drive robot controller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_node = my_robot_controller.teleop_node:main',
            'robot_controller = my_robot_controller.robot_controller:main',
            'odometry_node = my_robot_controller.odometry_node:main',
            'state_monitor = my_robot_controller.state_monitor:main',
        ],
    },
)
```

## مرحلہ 8: بلڈ اور رن کریں

### بلڈ کریں

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_controller my_robot_interfaces
source install/setup.bash
```

### سسٹم رن کریں

```bash
# Launch all nodes
ros2 launch my_robot_controller robot_system.launch.py

# In another terminal, run teleop
ros2 run my_robot_controller teleop_node

# In another terminal, monitor topics
ros2 topic echo /robot_status
```

### Emergency Stop ٹیسٹ کریں

```bash
# Activate emergency stop
ros2 service call /emergency_stop std_srvs/srv/SetBool "{data: true}"

# Deactivate emergency stop
ros2 service call /emergency_stop std_srvs/srv/SetBool "{data: false}"
```

## ویژولائزیشن

### RViz Configuration

```bash
rviz2
```

شامل کریں:
-   `/odom` کے لیے **Odometry** ڈسپلے
-   کوآرڈینیٹ فریمز کے لیے **TF** ڈسپلے
-   روبوٹ کی ٹریجیکٹری دکھانے کے لیے **Path** ڈسپلے

## ٹیسٹنگ اور تصدیق

### یونٹ ٹیسٹ (Placeholder)

```python
# test/test_robot_controller.py

import unittest
import rclpy
from my_robot_controller.robot_controller import RobotController

class TestRobotController(unittest.TestCase):
    def test_velocity_clamping(self):
        # Test that velocities are clamped to safe limits
        pass

    def test_emergency_stop(self):
        # Test emergency stop functionality
        pass

if __name__ == '__main__':
    unittest.main()
```

## اہم نکات

-   مناسب فن تعمیر کے ساتھ ایک ملٹی-نوڈ روبوٹ سسٹم بنایا۔
-   ڈومین-مخصوص ڈیٹا کے لیے کسٹم Messages کو نافذ کیا۔
-   حفاظتی اعتبار سے اہم فنکشنز کے لیے Services بنائیں۔
-   متعدد Nodes کو منظم کرنے کے لیے Launch files کا استعمال کیا۔
-   QoS پالیسیاں مناسب طریقے سے لاگو کیں۔
-   Odometry کیلکولیشن کو نافذ کیا۔
-   ویژولائزیشن کے لیے TF transforms کو مربوط کیا۔

## اگلے اقدامات

اب جبکہ آپ نے اپنی پہلی مکمل روبوٹ ایپلیکیشن بنا لی ہے، آئیے ROS 2 کے جدید تصورات کو دریافت کریں جن میں Actions، Lifecycle Nodes، اور Component Composition شامل ہیں!

---

**Previous:** [Nodes, Topics, and Services](./nodes-topics-services)
**Next:** [Advanced ROS 2 Concepts](./advanced-ros2)
