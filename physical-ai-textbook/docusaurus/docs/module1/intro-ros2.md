---
sidebar_position: 4
title: Introduction to ROS 2
description: Understanding the Robot Operating System and its architecture
---

# Introduction to ROS 2

## Overview

The Robot Operating System (ROS) is not an operating system in the traditional sense, but rather a middleware framework that provides a structured communication layer between robot software components. ROS 2, the second generation, brings significant improvements in real-time performance, security, and multi-robot systems.

## What is ROS 2?

ROS 2 is a set of software libraries and tools that help you build robot applications. It provides:

- **Hardware abstraction**: Interface with sensors and actuators uniformly
- **Low-level device control**: Direct communication with robot hardware
- **Message-passing between processes**: Standardized inter-process communication
- **Package management**: Organize and share robot software
- **Visualization and debugging tools**: Inspect and debug robot behavior

## Why ROS 2 Over ROS 1?

### Key Improvements

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| **Real-time support** | Limited | Native RTOS support |
| **Security** | None | DDS security plugins |
| **Communication** | Custom TCP/UDP | DDS middleware |
| **Multi-robot** | Challenging | Built-in support |
| **Cross-platform** | Mainly Linux | Linux, Windows, macOS |
| **Production ready** | Research focus | Industry-ready |

### When to Use ROS 2

✅ **Use ROS 2 for:**
- New robot projects
- Real-time control requirements
- Multi-robot systems
- Production deployments
- Cross-platform development

❌ **Consider ROS 1 only if:**
- Maintaining legacy systems
- Specific package unavailable in ROS 2

## ROS 2 Architecture

### Core Concepts

```
┌─────────────────────────────────────────────────┐
│                 Application Layer                │
│  (Your Robot Software - Nodes, Logic, Control)  │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                  ROS 2 Client Layer              │
│     (rclpy, rclcpp - ROS 2 Client Libraries)    │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                   ROS 2 Middleware               │
│         (rcl - ROS Client Library Core)         │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                  DDS Implementation              │
│    (FastDDS, CycloneDDS, RTI Connext, etc.)     │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                   Network Layer                  │
│              (UDP/TCP Transport)                 │
└─────────────────────────────────────────────────┘
```

### Communication Patterns

ROS 2 supports multiple communication patterns:

#### 1. Topics (Publish-Subscribe)
```
Publisher Node → Topic → Subscriber Node(s)
   (1-to-many, asynchronous)
```

#### 2. Services (Request-Response)
```
Client Node ↔ Service Node
   (1-to-1, synchronous)
```

#### 3. Actions (Goal-Based)
```
Action Client → Action Server
   (with feedback and cancellation)
```

#### 4. Parameters
```
Node ↔ Parameter Server
   (configuration management)
```

## DDS: The Foundation

### What is DDS?

Data Distribution Service (DDS) is a middleware protocol that ROS 2 uses for communication. It provides:

- **Discovery**: Automatic node discovery (no master node needed!)
- **Reliability**: Quality of Service (QoS) policies
- **Real-time**: Deterministic communication
- **Scalability**: Efficient multi-robot communication

### QoS Policies

Quality of Service policies control communication behavior:

| Policy | Options | Use Case |
|--------|---------|----------|
| **Reliability** | Best effort / Reliable | Sensor data vs. commands |
| **Durability** | Volatile / Transient local | Real-time vs. historical |
| **History** | Keep last / Keep all | Buffer management |
| **Deadline** | Duration | Real-time constraints |

## ROS 2 Distributions

ROS 2 follows a time-based release schedule:

| Distribution | Release Date | EOL | Status |
|--------------|--------------|-----|--------|
| **Foxy Fitzroy** | June 2020 | May 2023 | EOL |
| **Galactic Geochelone** | May 2021 | Nov 2022 | EOL |
| **Humble Hawksbill** | May 2022 | May 2027 | **LTS** ✅ |
| **Iron Irwini** | May 2023 | Nov 2024 | Supported |
| **Jazzy Jalisco** | May 2024 | May 2026 | Latest |

**This course uses ROS 2 Humble (LTS)**

## Installation Verification

Let's verify your ROS 2 installation:

```bash
# Check ROS 2 version
ros2 --version

# List installed packages
ros2 pkg list

# Check environment variables
printenv | grep ROS
```

Expected output:
```
ros2 cli version: 0.25.x
ROS_VERSION=2
ROS_PYTHON_VERSION=3
ROS_DISTRO=humble
```

## Your First ROS 2 Command

### List Running Nodes

```bash
ros2 node list
```

### Run a Demo Node

```bash
# Terminal 1: Run talker
ros2 run demo_nodes_cpp talker

# Terminal 2: Run listener
ros2 run demo_nodes_py listener
```

You should see messages being published and received!

### Inspect Topics

```bash
# List all topics
ros2 topic list

# Show message being published
ros2 topic echo /chatter

# Get topic info
ros2 topic info /chatter
```

## ROS 2 Workspace Structure

A typical ROS 2 workspace:

```
ros2_ws/                  # Workspace root
├── src/                  # Source space (your code)
│   ├── package_1/
│   │   ├── package.xml
│   │   ├── CMakeLists.txt (C++)
│   │   ├── setup.py (Python)
│   │   └── ...
│   └── package_2/
├── build/                # Build space (auto-generated)
├── install/              # Install space (executables)
└── log/                  # Build and runtime logs
```

## Core ROS 2 Concepts Summary

### Nodes
Independent processes that perform computation. Nodes communicate via topics, services, and actions.

### Packages
Organizational units for ROS 2 code. Contains related nodes, libraries, configs, and more.

### Topics
Named buses where nodes exchange messages. Publish-subscribe pattern.

### Messages
Data structures sent between nodes. Defined in `.msg` files.

### Services
Synchronous request-response communication between nodes.

### Actions
Long-running tasks with feedback, result, and goal cancellation.

## Development Tools

### Command Line Interface (CLI)

```bash
ros2 <verb> <sub-command> [options]
```

Common verbs:
- `node` - Node management
- `topic` - Topic inspection
- `service` - Service calls
- `action` - Action management
- `param` - Parameter operations
- `pkg` - Package operations
- `launch` - Launch file execution

### RQt Tools

ROS 2 provides graphical tools:

```bash
# Node graph visualization
rqt_graph

# Topic plotter
rqt_plot

# General purpose GUI
rqt
```

### Debugging

```bash
# Console output filtering
ros2 run --prefix 'gdb -ex run --args' <package> <executable>

# Launch file debugging
ros2 launch --debug <package> <launch_file>
```

## Best Practices from the Start

1. **Use descriptive names**: `/robot/sensor/lidar` not `/lidar1`
2. **Namespace your nodes**: Avoid global namespace pollution
3. **Follow naming conventions**: `snake_case` for topics, `CamelCase` for nodes
4. **Document your packages**: Write clear README files
5. **Use version control**: Git from day one
6. **Test incrementally**: Build and test often

## Common Gotchas

❗ **Forgetting to source**: Always `source install/setup.bash` after building
❗ **Mismatched QoS**: Publisher and subscriber QoS must be compatible
❗ **Circular dependencies**: Keep package dependencies acyclic
❗ **Stale builds**: Run `colcon build --cmake-clean-cache` when in doubt

## Next Steps

Now that you understand ROS 2 fundamentals, let's dive into creating nodes and setting up communication patterns in the next chapter!

## Key Takeaways

- ROS 2 is a middleware framework for building robot software
- DDS provides the underlying communication infrastructure
- QoS policies give fine-grained control over communication
- Nodes communicate via topics, services, and actions
- ROS 2 Humble (LTS) is production-ready and well-supported

## Additional Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Design](https://design.ros2.org/)
- [DDS Foundation](https://www.dds-foundation.org/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)

---

**Previous:** [Development Environment Setup](../intro/environment-setup)
**Next:** [Nodes, Topics, and Services](./nodes-topics-services)
