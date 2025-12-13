---
sidebar_position: 15
title: Isaac Gym for Reinforcement Learning
description: Massively parallel RL training with Isaac Gym
---

# Isaac Gym for Reinforcement Learning

## Introduction

Isaac Gym enables training RL policies with thousands of parallel environments running on GPU, dramatically accelerating training.

## Architecture

```
┌─────────────────────────────────────────┐
│         Python RL Algorithm             │
│      (PPO, SAC, TD3, etc.)             │
└─────────────────────────────────────────┘
                 ↕
┌─────────────────────────────────────────┐
│      Isaac Gym Vectorized Env           │
│   (1000s of parallel environments)      │
└─────────────────────────────────────────┘
                 ↕
┌─────────────────────────────────────────┐
│          GPU Physics (PhysX)            │
│     All envs simulated in parallel      │
└─────────────────────────────────────────┘
```

## Creating an RL Task

### Task Template

```python
from omni.isaac.gym.vec_env import VecEnvBase
import torch

class CartpoleTask(VecEnvBase):
    def __init__(self, name, num_envs=1024):
        super().__init__(name, num_envs=num_envs)

        # Define observation and action spaces
        self.observation_space = 4  # [x, x_dot, theta, theta_dot]
        self.action_space = 1       # force

    def create_envs(self):
        """Create parallel environments"""
        for i in range(self.num_envs):
            # Spawn cartpole at different positions
            pass

    def get_observations(self):
        """Return observations for all envs"""
        # Shape: (num_envs, obs_dim)
        return torch.zeros((self.num_envs, 4), device="cuda")

    def pre_physics_step(self, actions):
        """Apply actions before physics step"""
        # actions shape: (num_envs, action_dim)
        pass

    def post_physics_step(self):
        """Compute rewards after physics step"""
        # Calculate rewards
        rewards = torch.zeros(self.num_envs, device="cuda")

        # Check for episode termination
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda")

        return rewards, dones

    def reset(self):
        """Reset environments"""
        pass
```

## Training Loop

### PPO Training Example

```python
import torch
from torch import nn

class PPOAgent:
    def __init__(self, obs_dim, act_dim):
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        ).cuda()

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).cuda()

def train(task, agent, num_iterations=1000):
    for iteration in range(num_iterations):
        # Collect rollouts
        obs = task.get_observations()
        actions = agent.actor(obs)

        # Step environment
        task.pre_physics_step(actions)
        task.world.step()
        rewards, dones = task.post_physics_step()

        # Update policy
        # [PPO update logic]

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Mean Reward: {rewards.mean():.2f}")
```

## Domain Randomization

```python
def randomize_environment(task):
    """Randomize physics parameters"""

    # Randomize masses
    task.robot.set_masses(
        torch.rand(task.num_envs, device="cuda") * 2.0 + 0.5
    )

    # Randomize friction
    task.set_friction_coefficients(
        torch.rand(task.num_envs, device="cuda") * 0.5 + 0.5
    )

    # Randomize motor strength
    task.set_motor_gains(
        torch.rand(task.num_envs, device="cuda") * 20.0 + 10.0
    )
```

## Example Tasks

### 1. Cartpole Balancing
- **Observation**: Cart position, velocity, pole angle, angular velocity
- **Action**: Force applied to cart
- **Reward**: Time balanced

### 2. Robot Arm Reaching
- **Observation**: Joint positions, end-effector pose, target pose
- **Action**: Joint torques
- **Reward**: Negative distance to target

### 3. Quadruped Locomotion
- **Observation**: Joint states, IMU, contact sensors
- **Action**: Joint torques/positions
- **Reward**: Forward velocity, stability, energy efficiency

## Performance Tips

- Use GPU tensors throughout (avoid CPU-GPU transfers)
- Batch operations across environments
- Limit rendering during training (headless mode)
- Use appropriate number of parallel envs (1024-4096 typical)

## Key Takeaways

- Isaac Gym enables 1000s of parallel RL environments
- All computation on GPU for maximum speed
- Domain randomization built-in
- Dramatically faster than CPU-based training

---

**Previous:** [Isaac Sim Fundamentals](./isaac-sim-fundamentals)
**Next:** [Isaac ROS Integration](./isaac-ros)
