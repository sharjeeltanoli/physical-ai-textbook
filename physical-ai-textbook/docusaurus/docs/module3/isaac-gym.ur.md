---
sidebar_position: 15
title: Isaac Gym for Reinforcement Learning
description: Massively parallel RL training with Isaac Gym
---

# Isaac Gym برائے Reinforcement Learning

## تعارف

Isaac Gym RL پالیسیوں کی تربیت کو ممکن بناتا ہے جو GPU پر ہزاروں متوازی ماحول میں چلتی ہیں، اور تربیت کو ڈرامائی طور پر تیز کرتی ہے۔

## فن تعمیر

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

## ایک RL ٹاسک بنانا

### ٹاسک ٹیمپلیٹ

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

## تربیتی لوپ

### PPO تربیت کی مثال

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

## ڈومین رینڈمائزیشن

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

## مثالی ٹاسکس

### 1. کارٹپول بیلنسنگ
- **مشاہدہ**: کارٹ کی پوزیشن، رفتار، پول کا زاویہ، زاویائی رفتار
- **عمل**: کارٹ پر لگائی گئی قوت
- **انعام**: متوازن رہنے کا وقت

### 2. روبوٹ آرم کا پہنچنا
- **مشاہدہ**: جوائنٹ پوزیشنز، اینڈ-ایفیکٹر پوز، ٹارگٹ پوز
- **عمل**: جوائنٹ ٹارکس
- **انعام**: ٹارگٹ سے منفی فاصلہ

### 3. کواڈروپیڈ لوکوموشن
- **مشاہدہ**: جوائنٹ اسٹیٹس، IMU، کانٹیکٹ سینسرز
- **عمل**: جوائنٹ ٹارکس/پوزیشنز
- **انعام**: فارورڈ ویلوسٹی، استحکام، توانائی کی کارکردگی

## کارکردگی کے نکات

- GPU ٹینسرز کا ہر جگہ استعمال کریں (CPU-GPU منتقلی سے گریز کریں)
- ماحول میں بیچ آپریشنز کریں
- تربیت کے دوران رینڈرنگ کو محدود کریں (ہیڈلیس موڈ)
- متوازی ماحول کی مناسب تعداد استعمال کریں (عام طور پر 1024-4096)

## اہم نکات

- Isaac Gym ہزاروں متوازی RL ماحول کو ممکن بناتا ہے
- زیادہ سے زیادہ رفتار کے لیے تمام حساب کتاب GPU پر ہوتا ہے
- ڈومین رینڈمائزیشن بلٹ اِن ہے
- CPU پر مبنی تربیت سے ڈرامائی طور پر تیز ہے

---

**پچھلا:** [Isaac Sim Fundamentals](./isaac-sim-fundamentals)
**اگلا:** [Isaac ROS Integration](./isaac-ros)
