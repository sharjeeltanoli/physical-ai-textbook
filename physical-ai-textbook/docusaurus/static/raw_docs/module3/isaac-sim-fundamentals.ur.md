---
sidebar_position: 14
title: Isaac Sim Fundamentals
description: Creating and simulating robots in Isaac Sim
---

# Isaac Sim کے بنیادی اصول

## منظر کی تخلیق

### منظر بنانا

python
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage

world = World(stage_units_in_meters=1.0)

# Add ground plane
world.scene.add_default_ground_plane()

# Import robot from USD
robot_path = add_reference_to_stage(
    "/path/to/robot.usd",
    prim_path="/World/Robot"
)

## روبوٹ کی درآمد اور ترتیب

### URDF سے USD میں تبدیلی

```python
import omni.isaac.urdf as urdf_interface

# Convert URDF to USD
urdf_interface.import_robot(
    urdf_path="/path/to/robot.urdf",
    usd_path="/path/to/robot.usd"
)
```

### روبوٹ کنٹرول

```python
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction

robot = world.scene.add(Robot(prim_path="/World/Robot"))

# Apply joint commands
action = ArticulationAction(joint_positions=[0.0, 1.57, -1.57])
robot.apply_action(action)
```

## کیمرہ اور سینسرز

### RGB کیمرہ

```python
from omni.isaac.sensor import Camera

camera = Camera(
    prim_path="/World/Camera",
    position=[2.0, 2.0, 2.0],
    resolution=(640, 480),
    frequency=30
)

# Get image
rgb = camera.get_rgba()
```

### LIDAR

```python
from omni.isaac.range_sensor import _range_sensor

lidar = _range_sensor.acquire_lidar_sensor_interface()
```

## فزکس کی ترتیب

```python
from pxr import PhysxSchema

# Set physics properties
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World"))
physx_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
physx_scene.CreateGravityMagnitudeAttr().Set(9.81)
```

## اہم نکات

- Isaac Sim جو USD فارمیٹ پر مبنی ہے
- GPU سے تیز کردہ فزکس اور رینڈرنگ
- روبوٹ اور سینسر کے جامع APIs
- Omniverse کے ساتھ گہرا انضمام

---

**پچھلا:** [NVIDIA Isaac کا تعارف](./intro-isaac)
**اگلا:** [Isaac Gym برائے RL](./isaac-gym)
