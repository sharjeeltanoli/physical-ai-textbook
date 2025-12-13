---
sidebar_position: 10
title: Gazebo Simulator Fundamentals
description: Master Gazebo for robot simulation with ROS 2
---

# Gazebo سمیولیٹر کے بنیادی اصول

## Gazebo کا تعارف

Gazebo ایک طاقتور اوپن سورس 3D روبوٹ سمیولیٹر ہے جو ROS 2 کے ساتھ بغیر کسی رکاوٹ کے مربوط ہوتا ہے۔ یہ حقیقت پسندانہ physics simulation، sensor modeling، اور visualization کی صلاحیتیں فراہم کرتا ہے۔

## Gazebo کا ڈھانچہ

### بنیادی اجزاء

┌──────────────────────────────────────┐
│        Gazebo Client (GUI)           │
│     (Visualization & User Input)     │
└──────────────────────────────────────┘
              ↕ (Network)
┌──────────────────────────────────────┐
│        Gazebo Server (gzserver)      │
│   (Physics, Sensors, World State)    │
└──────────────────────────────────────┘
              ↕
┌──────────────────────────────────────┐
│         Plugin System                │
│  (Custom Controllers, Sensors, etc.) │
└──────────────────────────────────────┘

## SDF ورلڈ فائلیں

### بنیادی ورلڈ کا ڈھانچہ

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include robot model -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

## URDF روبوٹ کی تفصیل

### مکمل روبوٹ کی مثال

```xml
<?xml version="1.0"?>
<robot name="differential_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.6 0.4 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.6 0.4 0.2"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.167" ixy="0.0" ixz="0.0"
               iyy="0.367" iyz="0.0"
               izz="0.467"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>

    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0026" ixy="0" ixz="0"
               iyy="0.0026" iyz="0"
               izz="0.005"/>
    </inertial>
  </link>

  <!-- Left Wheel Joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Right Wheel (symmetric to left) -->
  <!-- [Similar structure] -->

  <!-- Caster Wheel for stability -->
  <!-- [Caster wheel definition] -->

</robot>
```

## Gazebo پلگ انز

### Differential Drive پلگ ان

```xml
<gazebo>
  <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <update_rate>50</update_rate>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>
  </plugin>
</gazebo>
```

### Camera پلگ ان

```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>800</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>camera</namespace>
        <remapping>image_raw:=image_raw</remapping>
        <remapping>camera_info:=camera_info</remapping>
      </ros>
      <camera_name>camera1</camera_name>
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR پلگ ان

```xml
<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </ray>
    <plugin name="gazebo_ros_lidar" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/lidar</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Gazebo کے لیے لانچ فائلیں

### بنیادی Gazebo لانچ

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Package directories
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_robot_description = FindPackageShare('my_robot_description')

    # World file path
    world_file = PathJoinSubstitution([
        pkg_robot_description,
        'worlds',
        'my_world.world'
    ])

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_gazebo_ros,
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'world': world_file}.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_urdf_content}]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## سمیولیشن ماحول کی تعمیر

### اشیاء شامل کرنا

```xml
<!-- Obstacle -->
<model name="obstacle_1">
  <pose>2 2 0.5 0 0 0</pose>
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.2 0.2 1</ambient>
      </material>
    </visual>
  </link>
</model>
```

## جانچ اور ڈی بگنگ

### ماڈل کا معائنہ

```bash
# List models
gz model --list

# Check model info
gz model --model-name my_robot --info

# Pause/unpause simulation
gz world --pause 1
gz world --pause 0
```

### ٹاپکس کی نگرانی

```bash
# List Gazebo topics
gz topic --list

# Echo topic
gz topic --echo /gazebo/default/pose/info
```

## کارکردگی کی اصلاح

**میش کو سادہ بنانا:**
- کم پولی والے collision meshes استعمال کریں۔
- visual meshes کو تفصیلی لیکن معقول رکھیں۔

**Physics ٹوننگ:**
```xml
<physics type="ode">
  <max_step_size>0.005</max_step_size>  <!-- Larger = faster but less accurate -->
  <real_time_update_rate>200</real_time_update_rate>
  <max_contacts>10</max_contacts>
</physics>
```

## اہم نکات

- Gazebo جامع physics اور sensor simulation فراہم کرتا ہے۔
- SDF اور URDF دنیاؤں اور روبوٹس کی وضاحت کرتے ہیں۔
- پلگ انز ROS 2 انٹیگریشن کے لیے فعالیت کو بڑھاتے ہیں۔
- لانچ فائلیں پیچیدہ سمیولیشنز کو ترتیب دیتی ہیں۔
- کارکردگی کے لیے وفاداری (fidelity) اور رفتار کے درمیان سمجھوتے کی ضرورت ہوتی ہے۔

---

**پچھلا:** [Introduction to Robot Simulation](./intro-simulation)
**اگلا:** [Unity for Robotics](./unity-robotics)
