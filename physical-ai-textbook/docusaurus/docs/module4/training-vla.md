---
sidebar_position: 20
title: Training VLA Models
description: Practical guide to training vision-language-action models
---

# Training VLA Models

## Dataset Preparation

### Data Format

```python
# Episode structure
episode = {
    'images': [img_0, img_1, ..., img_T],          # (T, H, W, 3)
    'actions': [action_0, action_1, ..., action_T], # (T, action_dim)
    'instruction': "Pick up the red block",
    'robot_state': [state_0, ..., state_T],        # (T, state_dim)
    'success': True
}
```

### Data Collection

```python
import h5py

class RobotDatasetCollector:
    def __init__(self, save_path):
        self.save_path = save_path
        self.episodes = []

    def collect_episode(self, robot, instruction):
        """Teleoperate robot and record data"""
        episode_data = {
            'images': [],
            'actions': [],
            'robot_state': [],
            'instruction': instruction
        }

        while not done:
            # Capture state
            image = robot.get_camera_image()
            state = robot.get_state()

            # Get human action (teleoperation)
            action = get_human_action()

            # Store
            episode_data['images'].append(image)
            episode_data['actions'].append(action)
            episode_data['robot_state'].append(state)

            # Execute action
            robot.execute_action(action)

        self.episodes.append(episode_data)

    def save(self):
        """Save dataset to HDF5"""
        with h5py.File(self.save_path, 'w') as f:
            for i, episode in enumerate(self.episodes):
                grp = f.create_group(f'episode_{i}')
                grp.create_dataset('images', data=np.array(episode['images']))
                grp.create_dataset('actions', data=np.array(episode['actions']))
                grp.create_dataset('instruction', data=episode['instruction'])
```

### Data Loading

```python
import torch
from torch.utils.data import Dataset, DataLoader

class VLADataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # Load episodes
        with h5py.File(data_path, 'r') as f:
            self.num_episodes = len(f.keys())

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            episode = f[f'episode_{idx}']

            images = torch.tensor(episode['images'][:])
            actions = torch.tensor(episode['actions'][:])
            instruction = episode['instruction'][()].decode('utf-8')

            if self.transform:
                images = self.transform(images)

            return {
                'images': images,
                'actions': actions,
                'instruction': instruction
            }
```

## Training Loop

```python
def train_vla_model(model, train_loader, val_loader, num_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            images = batch['images'].cuda()
            actions = batch['actions'].cuda()
            instructions = batch['instruction']

            # Tokenize instructions
            text_tokens = tokenizer(instructions, return_tensors='pt', padding=True).input_ids.cuda()

            # Forward pass
            predicted_actions = model(images, text_tokens)

            # Compute loss
            loss = F.mse_loss(predicted_actions, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].cuda()
                actions = batch['actions'].cuda()
                instructions = batch['instruction']
                text_tokens = tokenizer(instructions, return_tensors='pt', padding=True).input_ids.cuda()

                predicted_actions = model(images, text_tokens)
                loss = F.mse_loss(predicted_actions, actions)
                val_losses.append(loss.item())

        # Logging
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vla_model.pth')

        scheduler.step()
```

## Advanced Training Techniques

### 1. Curriculum Learning

```python
class CurriculumTrainer:
    def __init__(self, model):
        self.model = model
        self.difficulty_stages = ['easy', 'medium', 'hard']
        self.current_stage = 0

    def train_stage(self, stage):
        """Train on tasks of specific difficulty"""
        dataset = get_dataset_for_stage(stage)
        train_model(self.model, dataset)

    def should_advance_stage(self, validation_score):
        """Check if ready for next difficulty"""
        threshold = [0.8, 0.85, 0.9][self.current_stage]
        return validation_score > threshold

    def train_curriculum(self):
        for stage_id, stage in enumerate(self.difficulty_stages):
            self.current_stage = stage_id
            print(f"Training on {stage} tasks")

            while True:
                self.train_stage(stage)
                score = evaluate(self.model, stage)

                if self.should_advance_stage(score):
                    print(f"Advancing from {stage}")
                    break
```

### 2. Online Fine-Tuning

```python
def online_finetuning(model, robot, num_iterations=1000):
    """Fine-tune model with real robot interaction"""

    replay_buffer = []

    for iteration in range(num_iterations):
        # Collect new data
        observation = robot.get_observation()
        action = model.predict(observation['image'], observation['instruction'])

        # Execute and observe outcome
        next_obs, reward, done, info = robot.step(action)

        # Store experience
        replay_buffer.append({
            'obs': observation,
            'action': action,
            'reward': reward,
            'success': info['success']
        })

        # Fine-tune on recent experience
        if len(replay_buffer) > 32:
            batch = sample_batch(replay_buffer, batch_size=32)
            finetune_step(model, batch)
```

### 3. Multi-Task Training

```python
task_datasets = {
    'pick': PickDataset(),
    'place': PlaceDataset(),
    'push': PushDataset(),
    'drawer': DrawerDataset()
}

def multitask_training(model, task_datasets, epochs=100):
    for epoch in range(epochs):
        for task_name, dataset in task_datasets.items():
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            for batch in loader:
                # Add task embedding
                task_embedding = get_task_embedding(task_name)

                predicted_actions = model(
                    batch['images'],
                    batch['instructions'],
                    task_embedding=task_embedding
                )

                loss = compute_loss(predicted_actions, batch['actions'])
                optimize(loss)
```

## Transfer Learning

### Pretrained Vision-Language Models

```python
from transformers import CLIPModel

# Load pretrained CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Use CLIP encoders as initialization
class VLAWithCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize with CLIP
        self.vision_encoder = clip_model.vision_model
        self.language_encoder = clip_model.text_model

        # Freeze early layers
        for param in list(self.vision_encoder.parameters())[:20]:
            param.requires_grad = False

        # Add action head
        self.action_decoder = ActionDecoder()
```

## Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    # Train model
    model = VLAModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_loss = train_and_validate(model, optimizer, batch_size)

    return val_loss

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)
```

## Key Takeaways

- Collect diverse, high-quality robot demonstration data
- Use behavior cloning as primary training objective
- Apply curriculum learning for complex tasks
- Fine-tune pretrained vision-language models
- Online fine-tuning improves real-world performance
- Multi-task training enhances generalization

---

**Previous:** [VLA Architecture & Theory](./vla-architecture)
**Next:** [Deploying VLA for Robotics](./deploying-vla)
