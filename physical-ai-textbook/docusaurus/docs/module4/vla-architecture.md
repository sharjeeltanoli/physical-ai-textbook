---
sidebar_position: 19
title: VLA Architecture & Theory
description: Deep dive into VLA model architectures and training methods
---

# VLA Architecture & Theory

## Transformer Foundation

### Self-Attention Mechanism

```python
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

## VLA Model Components

### 1. Vision Encoder

```python
class VisionEncoder(nn.Module):
    """Encode images to feature vectors"""

    def __init__(self, image_size=224, patch_size=16, dim=768):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(12)
        ])

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        return x
```

### 2. Language Encoder

```python
class LanguageEncoder(nn.Module):
    """Encode text instructions"""

    def __init__(self, vocab_size=50000, dim=768, max_len=512):
        super().__init__()

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(12)
        ])

    def forward(self, token_ids):
        # Token embedding
        x = self.token_embed(token_ids)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        return x
```

### 3. Multimodal Fusion

```python
class CrossModalAttention(nn.Module):
    """Fuse vision and language features"""

    def __init__(self, dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, vision_feats, language_feats):
        # Cross attention: language attends to vision
        attn_out, _ = self.cross_attn(
            query=language_feats,
            key=vision_feats,
            value=vision_feats
        )
        x = self.norm1(language_feats + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

### 4. Action Decoder

```python
class ActionDecoder(nn.Module):
    """Decode to robot actions"""

    def __init__(self, dim, action_dim):
        super().__init__()

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, action_dim)
        )

    def forward(self, fused_features):
        # Use CLS token or mean pooling
        global_feat = fused_features[:, 0]  # CLS token

        # Predict action
        action = self.action_head(global_feat)

        return action
```

## Complete VLA Model

```python
class VLAModel(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.fusion = CrossModalAttention(dim=768)
        self.action_decoder = ActionDecoder(dim=768, action_dim=action_dim)

    def forward(self, image, text_tokens, robot_state=None):
        # Encode modalities
        vision_feats = self.vision_encoder(image)
        language_feats = self.language_encoder(text_tokens)

        # Fuse
        fused_feats = self.fusion(vision_feats, language_feats)

        # Optionally concatenate robot state
        if robot_state is not None:
            robot_embedding = self.robot_encoder(robot_state)
            fused_feats = torch.cat([fused_feats, robot_embedding], dim=1)

        # Decode to action
        action = self.action_decoder(fused_feats)

        return action
```

## Training Objectives

### 1. Behavior Cloning

```python
def behavior_cloning_loss(predicted_actions, expert_actions):
    """Imitation learning objective"""
    return F.mse_loss(predicted_actions, expert_actions)
```

### 2. Contrastive Learning

```python
def contrastive_loss(vision_feats, language_feats, temperature=0.07):
    """Align vision and language representations"""

    # Normalize features
    vision_feats = F.normalize(vision_feats, dim=-1)
    language_feats = F.normalize(language_feats, dim=-1)

    # Similarity matrix
    logits = torch.matmul(vision_feats, language_feats.T) / temperature

    # Labels (diagonal = positive pairs)
    labels = torch.arange(len(vision_feats), device=logits.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss
```

### 3. Action Diffusion

```python
class DiffusionActionDecoder(nn.Module):
    """Generate actions via diffusion"""

    def __init__(self, action_dim, num_steps=100):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_steps

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim + 768 + 1, 512),  # action + condition + timestep
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, condition, num_samples=1):
        """Sample actions via denoising"""
        device = condition.device

        # Start from noise
        actions = torch.randn(num_samples, self.action_dim, device=device)

        # Denoise iteratively
        for t in reversed(range(self.num_steps)):
            t_embed = torch.full((num_samples, 1), t / self.num_steps, device=device)
            noise_pred = self.denoiser(torch.cat([actions, condition, t_embed], dim=-1))
            actions = actions - noise_pred * (1.0 / self.num_steps)

        return actions
```

## Training Strategies

### Multi-Task Learning

```python
tasks = [
    'pick_object',
    'place_object',
    'push_object',
    'open_drawer',
    'close_drawer'
]

# Sample batch from multiple tasks
for batch in dataloader:
    task_id = batch['task_id']
    images = batch['images']
    instructions = batch['instructions']
    actions = batch['actions']

    # Forward pass (task-conditioned)
    predicted_actions = model(images, instructions, task_id=task_id)

    # Compute loss
    loss = F.mse_loss(predicted_actions, actions)
```

### Data Augmentation

```python
import torchvision.transforms as T

augmentations = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomRotation(degrees=15),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
])

def augment_data(image, action):
    """Apply data augmentation"""
    augmented_image = augmentations(image)
    return augmented_image, action
```

## Evaluation Metrics

```python
def evaluate_vla(model, eval_env, num_episodes=100):
    """Evaluate VLA policy"""

    success_count = 0
    total_rewards = []

    for episode in range(num_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get action from model
            action = model.predict(obs['image'], obs['instruction'])

            # Step environment
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward

        if info['success']:
            success_count += 1

        total_rewards.append(episode_reward)

    success_rate = success_count / num_episodes
    mean_reward = np.mean(total_rewards)

    return {
        'success_rate': success_rate,
        'mean_reward': mean_reward
    }
```

## Key Takeaways

- VLA models built on transformer architecture
- Vision and language encoded separately then fused
- Action decoding via MLP or diffusion
- Training via behavior cloning and contrastive learning
- Multi-task learning improves generalization
- Data augmentation critical for robustness

---

**Previous:** [Introduction to VLA Models](./intro-vla)
**Next:** [Training VLA Models](./training-vla)
