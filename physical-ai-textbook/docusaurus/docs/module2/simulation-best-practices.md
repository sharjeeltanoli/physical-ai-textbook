---
sidebar_position: 12
title: Simulation Best Practices
description: Strategies for effective robot simulation and sim-to-real transfer
---

# Simulation Best Practices

## Overview

Effective simulation requires balancing accuracy, performance, and transferability to real robots. This chapter covers best practices for creating reliable simulations and bridging the reality gap.

## Model Fidelity

### Physical Properties

**Mass and Inertia**
```xml
<!-- Accurate inertial properties critical for dynamics -->
<inertial>
  <mass value="10.5"/>  <!-- Measure actual robot mass -->
  <inertia ixx="0.167" ixy="0.0" ixz="0.0"
           iyy="0.367" iyz="0.0"
           izz="0.467"/>  <!-- Calculate from CAD or measurements -->
</inertial>
```

**Friction Coefficients**
```python
# Test real materials for accurate values
friction_values = {
    'rubber_on_wood': 0.8,
    'metal_on_metal': 0.15,
    'plastic_on_carpet': 0.6
}
```

### Mesh Quality

**Visual vs Collision Meshes**
```
Visual Mesh: High polygon count, detailed
  - Used for rendering
  - Can be complex

Collision Mesh: Low polygon count, convex hulls
  - Used for physics
  - Must be simple for performance
```

**Best Practices:**
- Visual mesh: 1000-10000 triangles
- Collision mesh: &lt;100 triangles
- Use convex decomposition for complex shapes

## Sensor Modeling

### Realistic Sensor Noise

**Camera Noise**
```python
import numpy as np

def add_camera_noise(image, noise_level=0.01):
    """Add realistic camera noise"""
    # Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape)

    # Shot noise (depends on light intensity)
    shot_noise = np.random.poisson(image * 255) / 255

    # Combine
    noisy_image = image + noise + (shot_noise - image) * 0.1
    return np.clip(noisy_image, 0, 1)
```

**LIDAR Noise**
```python
def add_lidar_noise(ranges, stddev=0.01, outlier_prob=0.01):
    """Add realistic LIDAR noise"""
    # Gaussian noise
    noise = np.random.normal(0, stddev, len(ranges))
    noisy_ranges = ranges + noise

    # Random outliers
    outliers = np.random.random(len(ranges)) < outlier_prob
    noisy_ranges[outliers] = np.random.uniform(0.1, 30, np.sum(outliers))

    return noisy_ranges
```

### Sensor Latency

```python
from collections import deque
import time

class DelayedSensor:
    def __init__(self, delay_ms=50):
        self.delay = delay_ms / 1000.0
        self.buffer = deque()

    def add_measurement(self, measurement):
        """Store measurement with timestamp"""
        self.buffer.append((time.time(), measurement))

    def get_delayed_measurement(self):
        """Retrieve measurement with delay"""
        current_time = time.time()

        while self.buffer:
            timestamp, measurement = self.buffer[0]
            if current_time - timestamp >= self.delay:
                self.buffer.popleft()
                return measurement
            else:
                return None
        return None
```

## Sim-to-Real Transfer

### Domain Randomization

**Visual Randomization**
```python
class VisualRandomizer:
    def randomize_textures(self, objects):
        """Randomize object textures"""
        for obj in objects:
            obj.color = (
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1)
            )

    def randomize_lighting(self, lights):
        """Randomize lighting conditions"""
        for light in lights:
            light.intensity = random.uniform(0.5, 2.0)
            light.color_temperature = random.uniform(3000, 7000)

    def randomize_camera(self, camera):
        """Randomize camera properties"""
        camera.exposure = random.uniform(0.8, 1.2)
        camera.white_balance = random.uniform(0.9, 1.1)
        camera.add_motion_blur(random.uniform(0, 0.1))
```

**Physics Randomization**
```python
class PhysicsRandomizer:
    def randomize_robot_params(self, robot):
        """Randomize robot physical parameters"""
        # Mass uncertainty (±10%)
        robot.mass *= random.uniform(0.9, 1.1)

        # Friction uncertainty
        robot.wheel_friction *= random.uniform(0.8, 1.2)

        # Motor response
        robot.motor_damping *= random.uniform(0.9, 1.1)

        # Center of mass offset
        robot.com_offset = [
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01)
        ]

    def randomize_environment(self, world):
        """Randomize environment parameters"""
        # Ground friction
        world.ground_friction = random.uniform(0.6, 1.0)

        # Gravity (slight variations)
        world.gravity = random.uniform(-9.85, -9.77)

        # Air resistance
        world.air_damping = random.uniform(0.0, 0.05)
```

### System Identification

**Measure Real Robot Parameters**
```python
def identify_motor_model(robot):
    """
    Collect data from real robot to identify motor model

    Steps:
    1. Apply known voltage inputs
    2. Measure resulting velocities
    3. Fit model parameters
    """

    voltages = np.linspace(0, 12, 20)
    velocities = []

    for voltage in voltages:
        robot.set_motor_voltage(voltage)
        time.sleep(1.0)  # Wait for steady state
        vel = robot.measure_wheel_velocity()
        velocities.append(vel)

    # Fit linear model: vel = k * voltage
    k = np.polyfit(voltages, velocities, 1)[0]

    return k

def update_simulation_model(sim_robot, identified_params):
    """Update simulation with identified parameters"""
    sim_robot.motor_constant = identified_params['motor_constant']
    sim_robot.friction_coefficient = identified_params['friction']
```

### Progressive Training

```python
class ProgressiveTrainer:
    """
    Train policy with gradually increasing realism

    Stage 1: Simple physics, no noise
    Stage 2: Realistic physics, minimal noise
    Stage 3: Full randomization
    """

    def __init__(self):
        self.stage = 1

    def get_randomization_level(self):
        if self.stage == 1:
            return {
                'noise_level': 0.0,
                'physics_randomization': False,
                'visual_randomization': False
            }
        elif self.stage == 2:
            return {
                'noise_level': 0.01,
                'physics_randomization': True,
                'visual_randomization': False
            }
        else:  # Stage 3
            return {
                'noise_level': 0.05,
                'physics_randomization': True,
                'visual_randomization': True
            }

    def advance_stage(self):
        """Move to next training stage"""
        if self.stage < 3:
            self.stage += 1
```

## Validation and Testing

### Sim-vs-Real Comparison

```python
def compare_trajectories(sim_traj, real_traj, plot=True):
    """
    Compare simulated and real robot trajectories

    Args:
        sim_traj: [(x, y, theta), ...]
        real_traj: [(x, y, theta), ...]
    """

    # Calculate errors
    position_errors = []
    for (x_sim, y_sim, _), (x_real, y_real, _) in zip(sim_traj, real_traj):
        error = np.sqrt((x_sim - x_real)**2 + (y_sim - y_real)**2)
        position_errors.append(error)

    # Metrics
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    final_error = position_errors[-1]

    print(f"Mean Error: {mean_error:.3f}m")
    print(f"Max Error: {max_error:.3f}m")
    print(f"Final Error: {final_error:.3f}m")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        # Trajectory plot
        plt.subplot(1, 2, 1)
        sim_x = [p[0] for p in sim_traj]
        sim_y = [p[1] for p in sim_traj]
        real_x = [p[0] for p in real_traj]
        real_y = [p[1] for p in real_traj]

        plt.plot(sim_x, sim_y, 'b-', label='Simulation')
        plt.plot(real_x, real_y, 'r--', label='Real')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.title('Trajectory Comparison')

        # Error plot
        plt.subplot(1, 2, 2)
        plt.plot(position_errors)
        plt.xlabel('Time Step')
        plt.ylabel('Position Error (m)')
        plt.title('Position Error Over Time')

        plt.tight_layout()
        plt.show()

    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'final_error': final_error
    }
```

### Automated Testing

```python
import unittest

class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator()
        self.robot = self.sim.spawn_robot()

    def test_forward_motion(self):
        """Test robot moves forward when commanded"""
        initial_pos = self.robot.get_position()

        # Command forward motion
        self.robot.set_velocity(linear=1.0, angular=0.0)
        self.sim.step(duration=1.0)

        final_pos = self.robot.get_position()

        # Check robot moved forward
        self.assertGreater(final_pos[0], initial_pos[0])
        self.assertAlmostEqual(final_pos[1], initial_pos[1], places=2)

    def test_rotation(self):
        """Test robot rotates when commanded"""
        initial_theta = self.robot.get_orientation()

        # Command rotation
        self.robot.set_velocity(linear=0.0, angular=1.0)
        self.sim.step(duration=1.0)

        final_theta = self.robot.get_orientation()

        # Check robot rotated
        self.assertNotAlmostEqual(final_theta, initial_theta, places=1)

    def test_collision_detection(self):
        """Test robot detects collisions"""
        # Place obstacle in front of robot
        self.sim.spawn_obstacle(position=(1, 0, 0))

        # Command forward motion
        self.robot.set_velocity(linear=1.0, angular=0.0)
        self.sim.step(duration=2.0)

        # Check collision detected
        self.assertTrue(self.robot.has_collision())
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_simulation():
    """Profile simulation performance"""

    profiler = cProfile.Profile()
    profiler.enable()

    # Run simulation
    sim = Simulator()
    for _ in range(1000):
        sim.step()

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

### Optimization Strategies

**1. Reduce Physics Complexity**
```python
# Use simplified collision shapes
collision_shape = 'box'  # Instead of detailed mesh

# Reduce contact points
max_contacts = 5

# Larger timestep (less accurate but faster)
timestep = 0.01  # Instead of 0.001
```

**2. LOD (Level of Detail)**
```python
class LODManager:
    def update_lod(self, objects, camera_pos):
        """Adjust detail based on distance from camera"""
        for obj in objects:
            distance = np.linalg.norm(obj.position - camera_pos)

            if distance < 5:
                obj.set_lod('high')
            elif distance < 20:
                obj.set_lod('medium')
            else:
                obj.set_lod('low')
```

**3. Parallel Simulation**
```python
from multiprocessing import Pool

def run_parallel_simulations(num_sims=10):
    """Run multiple simulations in parallel"""

    def run_single_sim(sim_id):
        sim = Simulator(seed=sim_id)
        robot = sim.spawn_robot()
        # Run simulation...
        return sim.get_results()

    with Pool(processes=num_sims) as pool:
        results = pool.map(run_single_sim, range(num_sims))

    return results
```

## Continuous Integration

### Automated Sim Testing

```yaml
# .github/workflows/simulation_tests.yml

name: Simulation Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install ROS 2
        run: |
          # Install ROS 2 Humble
          ...

      - name: Install Gazebo
        run: |
          sudo apt install gazebo
          ...

      - name: Build workspace
        run: |
          colcon build

      - name: Run simulation tests
        run: |
          source install/setup.bash
          pytest tests/simulation/
```

## Documentation

### Simulation Environment Spec

```markdown
# Simulation Environment Specification

## Robot Model
- **Model**: DifferentialDriveRobot v1.0
- **Mass**: 10.5 kg
- **Wheel diameter**: 0.2 m
- **Wheel separation**: 0.4 m
- **Max speed**: 2.0 m/s

## Sensors
### Camera
- **Resolution**: 640x480
- **FPS**: 30
- **FOV**: 80°
- **Noise**: Gaussian σ=0.01

### LIDAR
- **Range**: 0.1-30 m
- **Resolution**: 0.01 m
- **Scan rate**: 10 Hz
- **Angular resolution**: 1°

## Environment
- **Terrain**: Flat, friction=0.8
- **Obstacles**: Static boxes (1x1x1 m)
- **Lighting**: Directional sun light

## Physics
- **Engine**: ODE
- **Timestep**: 0.001 s
- **Gravity**: -9.81 m/s²
```

## Key Takeaways

- Accurate physical parameters critical for simulation fidelity
- Sensor noise modeling essential for robust algorithms
- Domain randomization bridges sim-to-real gap
- System identification calibrates simulation to real robot
- Validation through sim-vs-real comparison mandatory
- Performance optimization through LOD and parallelization
- Automated testing ensures simulation reliability
- Comprehensive documentation enables reproducibility

## Next Module

Congratulations on completing Module 2! Next, we'll explore NVIDIA Isaac platform for GPU-accelerated simulation and AI training!

---

**Previous:** [Unity for Robotics](./unity-robotics)
**Next:** [Module 3: Introduction to NVIDIA Isaac](../module3/intro-isaac)
