---
sidebar_position: 5
title: ROS 2 Nodes, Topics, and Services
description: Deep dive into ROS 2 communication patterns and building interactive systems
---

# ROS 2 Nodes, Topics, and Services

## Introduction

In this chapter, we'll explore the fundamental building blocks of ROS 2 applications: nodes, topics, and services. You'll learn how to create nodes, publish and subscribe to topics, and implement service-based communication.

## Understanding Nodes

### What is a Node?

A **node** is a single, modular process that performs a specific task in your robot system. Think of nodes as specialized workers in a factory:

- **Camera node**: Captures and publishes images
- **Motor controller node**: Receives commands and controls motors
- **Navigation node**: Plans paths and publishes velocity commands
- **Sensor fusion node**: Combines multiple sensor inputs

### Node Design Principles

✅ **Single Responsibility**: Each node should do one thing well
✅ **Loose Coupling**: Nodes communicate via standard interfaces
✅ **Reusability**: Well-designed nodes work across different robots
✅ **Testability**: Isolated nodes are easier to test

## Creating Your First Node

### Python Node

```python
# ~/ros2_ws/src/my_robot_controller/my_robot_controller/my_first_node.py

import rclpy
from rclpy.node import Node

class MyFirstNode(Node):
    def __init__(self):
        super().__init__('my_first_node')
        self.get_logger().info('Hello from ROS 2!')

        # Create a timer that calls a function every second
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        self.counter += 1
        self.get_logger().info(f'Timer callback {self.counter}')

def main(args=None):
    rclpy.init(args=args)
    node = MyFirstNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### C++ Node

```cpp
// ~/ros2_ws/src/my_robot_controller/src/my_first_node.cpp

#include "rclcpp/rclcpp.hpp"

class MyFirstNode : public rclcpp::Node {
public:
    MyFirstNode() : Node("my_first_node"), counter_(0) {
        RCLCPP_INFO(this->get_logger(), "Hello from ROS 2!");

        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MyFirstNode::timer_callback, this)
        );
    }

private:
    void timer_callback() {
        counter_++;
        RCLCPP_INFO(this->get_logger(), "Timer callback %d", counter_);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MyFirstNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

### Node Lifecycle

```
┌──────────────────────────────────────────┐
│           Unconfigured State             │
└──────────────────────────────────────────┘
                    ↓ configure()
┌──────────────────────────────────────────┐
│            Inactive State                │
└──────────────────────────────────────────┘
                    ↓ activate()
┌──────────────────────────────────────────┐
│             Active State                 │
└──────────────────────────────────────────┘
                    ↓ deactivate()
┌──────────────────────────────────────────┐
│            Inactive State                │
└──────────────────────────────────────────┘
                    ↓ cleanup()
┌──────────────────────────────────────────┐
│         Unconfigured State               │
└──────────────────────────────────────────┘
```

## Topics: Publish-Subscribe Pattern

### What are Topics?

Topics are named channels for streaming data. They implement the publish-subscribe pattern:

- **Publishers** send messages to topics
- **Subscribers** receive messages from topics
- **Many-to-many**: Multiple publishers and subscribers per topic

### Creating a Publisher

#### Python Publisher

```python
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')

        # Create publisher
        self.publisher_ = self.create_publisher(
            String,           # Message type
            'chatter',        # Topic name
            10                # Queue size
        )

        # Publish every 0.5 seconds
        self.timer = self.create_timer(0.5, self.publish_message)
        self.counter = 0

    def publish_message(self):
        msg = String()
        msg.data = f'Hello ROS 2: {self.counter}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.counter += 1
```

#### C++ Publisher

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class PublisherNode : public rclcpp::Node {
public:
    PublisherNode() : Node("publisher_node"), counter_(0) {
        publisher_ = this->create_publisher<std_msgs::msg::String>(
            "chatter", 10
        );

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&PublisherNode::publish_message, this)
        );
    }

private:
    void publish_message() {
        auto msg = std_msgs::msg::String();
        msg.data = "Hello ROS 2: " + std::to_string(counter_);
        publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published: '%s'", msg.data.c_str());
        counter_++;
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;
};
```

### Creating a Subscriber

#### Python Subscriber

```python
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')

        # Create subscriber
        self.subscription = self.create_subscription(
            String,                    # Message type
            'chatter',                 # Topic name
            self.listener_callback,    # Callback function
            10                         # Queue size
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: "{msg.data}"')
```

#### C++ Subscriber

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class SubscriberNode : public rclcpp::Node {
public:
    SubscriberNode() : Node("subscriber_node") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "chatter",
            10,
            std::bind(&SubscriberNode::listener_callback, this, std::placeholders::_1)
        );
    }

private:
    void listener_callback(const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

### Custom Messages

Create custom message types for your specific needs.

#### Define Message

```bash
# ~/ros2_ws/src/my_robot_interfaces/msg/RobotStatus.msg

string robot_name
float32 battery_percentage
bool is_moving
int32 error_code
```

#### Use Custom Message

```python
from my_robot_interfaces.msg import RobotStatus

class StatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')
        self.publisher_ = self.create_publisher(
            RobotStatus,
            'robot_status',
            10
        )
        self.timer = self.create_timer(1.0, self.publish_status)

    def publish_status(self):
        msg = RobotStatus()
        msg.robot_name = "MyRobot"
        msg.battery_percentage = 75.5
        msg.is_moving = True
        msg.error_code = 0

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published status: {msg.battery_percentage}%')
```

## Services: Request-Response Pattern

### What are Services?

Services provide synchronous request-response communication:

- **Client** sends a request and waits for a response
- **Server** processes the request and sends back a response
- **One-to-one**: Each request has exactly one response

### Use Cases for Services

- Configuration changes
- Mathematical computations
- Starting/stopping processes
- Short-lived queries

### Creating a Service Server

#### Python Service Server

```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')

        # Create service
        self.service = self.create_service(
            AddTwoInts,                    # Service type
            'add_two_ints',                # Service name
            self.add_two_ints_callback     # Callback function
        )
        self.get_logger().info('Add Two Ints Server has been started')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response
```

#### C++ Service Server

```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class AddTwoIntsServer : public rclcpp::Node {
public:
    AddTwoIntsServer() : Node("add_two_ints_server") {
        service_ = this->create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints",
            std::bind(&AddTwoIntsServer::add_two_ints_callback,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2)
        );
        RCLCPP_INFO(this->get_logger(), "Add Two Ints Server has been started");
    }

private:
    void add_two_ints_callback(
        const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
        std::shared_ptr<example_interfaces::srv::AddTwoInts::Response> response
    ) {
        response->sum = request->a + request->b;
        RCLCPP_INFO(this->get_logger(), "%ld + %ld = %ld",
                    request->a, request->b, response->sum);
    }

    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};
```

### Creating a Service Client

#### Python Service Client

```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')

        # Create client
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.get_logger().info('Service is available!')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Call service asynchronously
        future = self.client.call_async(request)
        return future

# Usage
def main():
    rclpy.init()
    node = AddTwoIntsClient()
    future = node.send_request(5, 7)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        response = future.result()
        node.get_logger().info(f'Result: {response.sum}')

    rclpy.shutdown()
```

### Custom Services

#### Define Service

```bash
# ~/ros2_ws/src/my_robot_interfaces/srv/SetSpeed.srv

# Request
float32 linear_velocity
float32 angular_velocity
---
# Response
bool success
string message
```

#### Use Custom Service

```python
from my_robot_interfaces.srv import SetSpeed

class SpeedControlServer(Node):
    def __init__(self):
        super().__init__('speed_control_server')
        self.service = self.create_service(
            SetSpeed,
            'set_robot_speed',
            self.set_speed_callback
        )

    def set_speed_callback(self, request, response):
        # Validate speed limits
        if abs(request.linear_velocity) > 2.0:
            response.success = False
            response.message = "Linear velocity exceeds limit"
            return response

        # Apply speed (placeholder)
        self.get_logger().info(
            f'Setting speed: linear={request.linear_velocity}, '
            f'angular={request.angular_velocity}'
        )

        response.success = True
        response.message = "Speed set successfully"
        return response
```

## QoS (Quality of Service) Profiles

### Understanding QoS

QoS policies control communication behavior. Common profiles:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Sensor data (best effort, volatile)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=10
)

# Command messages (reliable, transient local)
command_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=10
)

# Create publisher with custom QoS
self.publisher_ = self.create_publisher(
    LaserScan,
    'scan',
    sensor_qos
)
```

### QoS Policy Examples

| Scenario | Reliability | Durability | Use Case |
|----------|-------------|------------|----------|
| **Sensor streams** | Best effort | Volatile | High-frequency data (IMU, camera) |
| **Commands** | Reliable | Transient local | Motor commands, goals |
| **Configuration** | Reliable | Transient local | Parameters, settings |
| **Logs** | Best effort | Volatile | Debug messages |

## Practical Example: Robot Odometry System

### Complete System Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Wheel Encoders │────────>│ Odometry Node    │
└─────────────────┘ /ticks  └──────────────────┘
                                     │ /odom
                                     ↓
                            ┌──────────────────┐
                            │ Navigation Node  │
                            └──────────────────┘
                                     │ /cmd_vel
                                     ↓
                            ┌──────────────────┐
                            │ Motor Controller │
                            └──────────────────┘
```

### Implementation Sketch

```python
# Odometry publisher node
class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Subscribe to wheel encoder ticks
        self.encoder_sub = self.create_subscription(
            EncoderTicks,
            'wheel_ticks',
            self.encoder_callback,
            10
        )

        # Publish odometry
        self.odom_pub = self.create_publisher(
            Odometry,
            'odom',
            10
        )

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def encoder_callback(self, msg):
        # Calculate odometry from encoder ticks
        # (simplified - real implementation more complex)
        delta_left = msg.left_ticks * 0.001  # meters per tick
        delta_right = msg.right_ticks * 0.001

        delta_s = (delta_right + delta_left) / 2.0
        delta_theta = (delta_right - delta_left) / 0.5  # wheel base

        self.x += delta_s * math.cos(self.theta)
        self.y += delta_s * math.sin(self.theta)
        self.theta += delta_theta

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        # ... (fill remaining fields)

        self.odom_pub.publish(odom_msg)
```

## Debugging Tools

### Command Line Tools

```bash
# List all topics
ros2 topic list

# Show topic info
ros2 topic info /chatter

# Echo topic messages
ros2 topic echo /chatter

# Check message frequency
ros2 topic hz /chatter

# Publish to topic manually
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"

# List services
ros2 service list

# Call service from CLI
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 5, b: 7}"

# Show service type
ros2 service type /add_two_ints
```

### RQt Graph

```bash
# Visualize node graph
rqt_graph
```

## Best Practices

### Topic Design

✅ **Use namespaces**: `/robot1/camera/image` not `/camera_image`
✅ **Descriptive names**: `/mobile_base/cmd_vel` not `/vel`
✅ **Standard message types**: Use existing types when possible
✅ **Appropriate QoS**: Match QoS to data characteristics

### Node Design

✅ **Single purpose**: Each node does one thing well
✅ **Composable**: Nodes can be combined in launch files
✅ **Parameterizable**: Use parameters for configuration
✅ **Error handling**: Gracefully handle failures

## Common Pitfalls

❌ **QoS mismatch**: Publisher and subscriber QoS incompatible
❌ **Blocking callbacks**: Long-running operations in callbacks
❌ **Memory leaks**: Not destroying nodes properly
❌ **Tight coupling**: Hard-coding topic names

## Next Steps

In the next chapter, we'll build a complete robot application that integrates multiple nodes, topics, and services!

## Key Takeaways

- Nodes are independent processes that perform specific tasks
- Topics enable asynchronous many-to-many communication
- Services provide synchronous request-response patterns
- QoS policies control communication reliability and durability
- Custom messages and services allow domain-specific interfaces

---

**Previous:** [Introduction to ROS 2](./intro-ros2)
**Next:** [Building Your First Robot Application](./first-robot-app)
