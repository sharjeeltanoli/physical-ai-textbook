---
sidebar_position: 17
title: Advanced Isaac Features
description: Advanced capabilities of the Isaac platform
---

# اعلیٰ Isaac خصوصیات

## مصنوعی ڈیٹا کی تخلیق

### Replicator API

python
import omni.replicator.core as rep

# Create randomizer
with rep.new_layer():
    # Randomize camera position
    camera = rep.create.camera(
        position=rep.distribution.uniform((-5, -5, 2), (5, 5, 5))
    )

    # Randomize object materials
    objects = rep.get.prims(semantics=[("class", "object")])
    with objects:
        rep.randomizer.materials(
            materials=rep.get.material_prims()
        )

    # Randomize lighting
    lights = rep.get.light_prims()
    with lights:
        rep.modify.attribute("intensity", rep.distribution.uniform(500, 2000))

# Setup writer for data output
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="/data/synthetic",
    rgb=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True
)

# Run data generation
rep.orchestrator.run()

### خودکار لیبلنگ

```python
# Semantic segmentation labels
rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

# Bounding boxes
rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")

# Instance segmentation
rep.AnnotatorRegistry.get_annotator("instance_segmentation")

# Depth
rep.AnnotatorRegistry.get_annotator("distance_to_camera")
```

## کثیر روبوٹ سمولیشن

```python
from omni.isaac.core import World

world = World()

# Spawn multiple robots
for i in range(10):
    robot = world.scene.add(
        Robot(
            prim_path=f"/World/Robot_{i}",
            name=f"robot_{i}",
            position=[i * 2.0, 0, 0]
        )
    )

# Coordinate robots
for i in range(1000):
    for robot in world.scene.get_all_robots():
        # Apply coordinated control
        pass
    world.step()
```

## اپنی مرضی کی فزکس

### آرٹیکولیشن کنفیگریشن

```python
from omni.isaac.core.articulations import Articulation

robot = Articulation(prim_path="/World/Robot")

# Set joint properties
robot.set_joint_positions([0, 0, 0, 0, 0, 0])
robot.set_joint_velocities([0, 0, 0, 0, 0, 0])
robot.set_joint_efforts([10, 10, 10, 10, 10, 10])

# Get robot state
positions = robot.get_joint_positions()
velocities = robot.get_joint_velocities()
```

### کانٹیکٹ سینسرز

```python
from omni.isaac.sensor import ContactSensor

contact_sensor = ContactSensor(
    prim_path="/World/Robot/gripper/contact_sensor",
    min_threshold=0.1,
    max_threshold=10.0
)

# Check contact
is_in_contact = contact_sensor.is_in_contact()
contact_force = contact_sensor.get_contact_force()
```

## Isaac Cortex سلوک کے درخت

### سلوک کی تعریف کریں

```python
from omni.isaac.cortex import Behavior, BehaviorNode

class PickAndPlace(Behavior):
    def __init__(self):
        super().__init__()

        # Define behavior tree
        self.root = BehaviorNode.Sequence([
            self.approach_object(),
            self.grasp_object(),
            self.move_to_target(),
            self.release_object()
        ])

    def approach_object(self):
        return BehaviorNode.Action(lambda: self.move_to(self.object_pos))

    def grasp_object(self):
        return BehaviorNode.Action(lambda: self.close_gripper())

    def move_to_target(self):
        return BehaviorNode.Action(lambda: self.move_to(self.target_pos))

    def release_object(self):
        return BehaviorNode.Action(lambda: self.open_gripper())
```

## Omniverse Nucleus باہمی تعاون

### اثاثے شیئر کریں

```bash
# Connect to Nucleus server
omniverse://localhost/Projects/

# Upload USD asset
omni.client.copy("file:///local/robot.usd", "omniverse://localhost/Projects/robot.usd")

# Collaborate in real-time
# Multiple users can edit the same scene simultaneously
```

## کارکردگی کی پروفائلنگ

### بلٹ ان پروفائلر

```python
import omni.kit.profiler as profiler

# Start profiling
profiler.begin_capture()

# Run simulation
for i in range(1000):
    world.step()

# Stop profiling
profiler.end_capture()
profiler.save_capture("/path/to/profile.json")
```

## کلاؤڈ تعیناتی

### کلاؤڈ پر Isaac Sim

```bash
# Launch on AWS/GCP/Azure with GPU instance
# NGC Container: nvcr.io/nvidia/isaac-sim

docker run --gpus all -it \
  -v /data:/data \
  nvcr.io/nvidia/isaac-sim:latest \
  /isaac-sim/isaac-sim.sh --headless
```

## اہم نکات

- مصنوعی ڈیٹا کی تخلیق کے لیے Replicator API
- کثیر روبوٹ سمولیشن کی صلاحیتیں
- پیچیدہ روبوٹ سلوک کے لیے سلوک کے درخت
- Omniverse باہمی تعاون کی خصوصیات
- اسکیل ایبلٹی کے لیے کلاؤڈ تعیناتی
- جامع پروفائلنگ ٹولز

## اگلا ماڈیول

ماڈیول 3 مکمل کرنے پر مبارکباد! اگلا، ہم قدرتی زبان سے چلنے والے روبوٹ کنٹرول کے لیے Vision-Language-Action ماڈلز کو تلاش کریں گے!

---

**پچھلا:** [Isaac ROS انٹیگریشن](./isaac-ros)
**اگلا:** [ماڈیول 4: VLA ماڈلز کا تعارف](../module4/intro-vla)
