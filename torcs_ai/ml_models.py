"""
Advanced Machine Learning Models for TORCS Racing AI

Uses PyTorch for deep neural networks and reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random

logger = logging.getLogger(__name__)

class RacingDataset(Dataset):
    """Dataset for racing experiences."""

    def __init__(self, experiences: List[Dict[str, Any]]):
        self.experiences = experiences

    def __len__(self) -> int:
        return len(self.experiences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        exp = self.experiences[idx]
        sensors = exp['sensors']
        actions = exp['actions']

        # Extract features
        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)
        track_pos = sensors.get('trackPos', 0)
        curvature = self._calculate_curvature(sensors)

        features = torch.tensor([
            speed, angle, track_pos, curvature,
            speed**2, angle**2, abs(track_pos), curvature*speed
        ], dtype=torch.float32)

        targets = torch.tensor([
            actions.get('steer', 0),
            actions.get('accel', 0),
            actions.get('brake', 0)
        ], dtype=torch.float32)

        return features, targets

    def _calculate_curvature(self, sensors: Dict[str, Any]) -> float:
        """Calculate track curvature from sensor data."""
        track = sensors.get('track', [])
        if len(track) < 5:
            return 0.0

        # Simple curvature estimation
        diffs = np.diff(track[:5])
        curvature = np.mean(np.abs(diffs))
        return float(curvature)


class RacingNetwork(nn.Module):
    """Neural network for racing action prediction."""

    def __init__(self, input_size: int = 8, hidden_size: int = 128, output_size: int = 3):
        super(RacingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent(nn.Module):
    """Deep Q-Network for reinforcement learning."""

    def __init__(self, state_size: int = 8, action_size: int = 9, hidden_size: int = 128):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLRacingAI:
    """Advanced Machine Learning-powered Racing AI with Deep Learning."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.data_collector = DataCollector()
        self.models = {}
        self.scalers = {}  # Keep for compatibility
        self.is_trained = False
        self.learning_mode = True

        # Neural networks
        self.steer_net = RacingNetwork(input_size=8, output_size=1).to(self.device)
        self.accel_net = RacingNetwork(input_size=8, output_size=1).to(self.device)
        self.brake_net = RacingNetwork(input_size=8, output_size=1).to(self.device)

        # DQN for reinforcement learning
        self.dqn_agent = DQNAgent().to(self.device)
        self.target_dqn = DQNAgent().to(self.device)
        self.target_dqn.load_state_dict(self.dqn_agent.state_dict())

        # Optimizers
        self.optimizers = {
            'steer': optim.Adam(self.steer_net.parameters(), lr=1e-4),
            'accel': optim.Adam(self.accel_net.parameters(), lr=1e-4),
            'brake': optim.Adam(self.brake_net.parameters(), lr=1e-4),
            'dqn': optim.Adam(self.dqn_agent.parameters(), lr=1e-4)
        }

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # State tracking
        self.previous_steer = 0
        self.previous_accel = 0.8
        self.previous_brake = 0
        self.prev_damage = 0
        self.prev_fuel = 100

        # Load existing models
        self.load_models()

    def load_models(self) -> None:
        """Load pre-trained models if available."""
        try:
            checkpoint = torch.load('racing_ai_models.pth', map_location=self.device)
            self.steer_net.load_state_dict(checkpoint['steer_net'])
            self.accel_net.load_state_dict(checkpoint['accel_net'])
            self.brake_net.load_state_dict(checkpoint['brake_net'])
            self.dqn_agent.load_state_dict(checkpoint['dqn_agent'])
            self.target_dqn.load_state_dict(checkpoint['target_dqn'])
            self.is_trained = True
            logger.info("Loaded pre-trained neural network models!")
        except FileNotFoundError:
            logger.info("No pre-trained models found, training initial models...")
            self.train_initial_models()

    def train_initial_models(self) -> None:
        """Train initial neural network models."""
        logger.info("Training initial neural network models...")

        # Generate synthetic racing data
        n_samples = 20000
        dataset = self._generate_synthetic_dataset(n_samples)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        # Train each network
        for action, net in [('steer', self.steer_net), ('accel', self.accel_net), ('brake', self.brake_net)]:
            logger.info(f"Training {action} network...")
            self._train_network(net, self.optimizers[action], dataloader, action, epochs=50)

        self.is_trained = True
        self.save_models()
        logger.info("Initial neural network models trained!")

    def _generate_synthetic_dataset(self, n_samples: int) -> RacingDataset:
        """Generate synthetic racing dataset."""
        experiences = []
        for _ in range(n_samples):
            speed = np.random.uniform(0, 320)
            angle = np.random.normal(0, 0.3)
            track_pos = np.random.normal(0, 0.5)
            curvature = np.random.uniform(0, 1)

            sensors = {
                'speedX': speed,
                'angle': angle,
                'trackPos': track_pos,
                'track': np.random.uniform(-1, 1, 19)
            }

            actions = {
                'steer': self.expert_steering(speed, angle, track_pos, curvature),
                'accel': self.expert_acceleration(speed, angle, curvature),
                'brake': self.expert_braking(speed, angle, curvature)
            }

            experiences.append({'sensors': sensors, 'actions': actions, 'reward': 0})

        return RacingDataset(experiences)

    def _train_network(self, net: nn.Module, optimizer: optim.Optimizer,
                      dataloader: DataLoader, action: str, epochs: int = 10) -> None:
        """Train a single neural network."""
        criterion = nn.MSELoss()
        net.train()

        for epoch in range(epochs):
            total_loss = 0
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets[:, {'steer': 0, 'accel': 1, 'brake': 2}[action]].unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = net(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def expert_steering(self, speed: float, angle: float, track_pos: float, curvature: float) -> float:
        """Generate expert steering decisions."""
        steer = angle * 25 / np.pi - track_pos * 0.25
        if curvature > 0.3:
            steer *= 1.2
        return np.clip(steer, -1, 1)

    def expert_acceleration(self, speed: float, angle: float, curvature: float) -> float:
        """Generate expert acceleration decisions."""
        target_speed = 280 * (1 - curvature * 0.4)
        speed_error = target_speed - speed

        if abs(angle) > 0.3:
            return 0.3
        elif speed_error > 20:
            return 1.0
        elif speed_error > 0:
            return 0.7
        else:
            return 0.2

    def expert_braking(self, speed: float, angle: float, curvature: float) -> float:
        """Generate expert braking decisions."""
        if abs(angle) > 0.4 or (speed > 250 and curvature > 0.4):
            return min(0.8, abs(angle) * 2)
        return 0.0

    def predict_action(self, sensor_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Use neural networks to predict optimal actions."""
        if not self.is_trained:
            return None

        speed = sensor_data.get('speedX', 0)
        angle = sensor_data.get('angle', 0)
        track_pos = sensor_data.get('trackPos', 0)
        curvature = self._calculate_curvature(sensor_data)

        features = torch.tensor([
            speed, angle, track_pos, curvature,
            speed**2, angle**2, abs(track_pos), curvature*speed
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        predictions = {}

        with torch.no_grad():
            self.steer_net.eval()
            self.accel_net.eval()
            self.brake_net.eval()

            steer_pred = self.steer_net(features).cpu().item()
            accel_pred = self.accel_net(features).cpu().item()
            brake_pred = self.brake_net(features).cpu().item()

            predictions['steer'] = np.clip(steer_pred, -1, 1)
            predictions['accel'] = np.clip(accel_pred, 0, 1)
            predictions['brake'] = np.clip(brake_pred, 0, 1)

        return predictions

    def _calculate_curvature(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate track curvature."""
        track = sensor_data.get('track', [])
        if len(track) < 5:
            return 0.0
        diffs = np.diff(track[:5])
        return float(np.mean(np.abs(diffs)))

    def update_models(self, sensor_data: Dict[str, Any], actions_taken: Dict[str, float], reward: float) -> None:
        """Update models with new experience."""
        if not self.learning_mode:
            return

        experience = {
            'state': self._extract_state(sensor_data),
            'action': self._discretize_actions(actions_taken),
            'reward': reward,
            'next_state': self._extract_state(sensor_data),  # Simplified
            'done': False
        }

        self.memory.append(experience)
        self.data_collector.add_experience({
            'sensors': sensor_data,
            'actions': actions_taken,
            'reward': reward,
            'timestamp': time.time()
        })

        if len(self.memory) > self.batch_size:
            self._train_dqn()

        # Retrain neural networks periodically
        if len(self.data_collector.experiences) % 1000 == 0:
            self.retrain_models()

    def _extract_state(self, sensor_data: Dict[str, Any]) -> torch.Tensor:
        """Extract state tensor from sensor data."""
        speed = sensor_data.get('speedX', 0)
        angle = sensor_data.get('angle', 0)
        track_pos = sensor_data.get('trackPos', 0)
        curvature = self._calculate_curvature(sensor_data)

        return torch.tensor([
            speed, angle, track_pos, curvature,
            speed**2, angle**2, abs(track_pos), curvature*speed
        ], dtype=torch.float32)

    def _discretize_actions(self, actions: Dict[str, float]) -> int:
        """Discretize continuous actions for DQN."""
        steer = int((actions.get('steer', 0) + 1) / 2 * 2)  # 0-2
        accel = int(actions.get('accel', 0) * 2)  # 0-2
        brake = int(actions.get('brake', 0) * 2)  # 0-2
        return steer * 9 + accel * 3 + brake

    def _train_dqn(self) -> None:
        """Train the DQN agent."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32).to(self.device)

        # Current Q values
        current_q = self.dqn_agent(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizers['dqn'].zero_grad()
        loss.backward()
        self.optimizers['dqn'].step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if len(self.memory) % 1000 == 0:
            self.target_dqn.load_state_dict(self.dqn_agent.state_dict())

    def retrain_models(self) -> None:
        """Retrain neural networks with accumulated experience."""
        if len(self.data_collector.experiences) < 100:
            return

        logger.info(f"Retraining neural networks with {len(self.data_collector.experiences)} experiences...")

        dataset = RacingDataset(self.data_collector.experiences[-5000:])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for action, net in [('steer', self.steer_net), ('accel', self.accel_net), ('brake', self.brake_net)]:
            self._train_network(net, self.optimizers[action], dataloader, action, epochs=5)

        self.save_models()
        logger.info("Neural networks retrained!")

    def save_models(self) -> None:
        """Save trained models."""
        try:
            torch.save({
                'steer_net': self.steer_net.state_dict(),
                'accel_net': self.accel_net.state_dict(),
                'brake_net': self.brake_net.state_dict(),
                'dqn_agent': self.dqn_agent.state_dict(),
                'target_dqn': self.target_dqn.state_dict(),
                'training_date': time.time()
            }, 'racing_ai_models.pth')
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def drive(self, c) -> None:
        """Main driving logic with neural networks."""
        S, R = c.S.d, c.R.d

        # Try neural network prediction
        nn_actions = self.predict_action(S)

        if nn_actions and random.random() > self.epsilon:
            # Use neural network predictions
            R['steer'] = nn_actions['steer']
            R['accel'] = nn_actions['accel']
            R['brake'] = nn_actions['brake']
        else:
            # Fallback to expert system
            R['steer'] = self.expert_steering(
                S.get('speedX', 0), S.get('angle', 0),
                S.get('trackPos', 0), self._calculate_curvature(S)
            )
            R['accel'] = self.expert_acceleration(
                S.get('speedX', 0), S.get('angle', 0), self._calculate_curvature(S)
            )
            R['brake'] = self.expert_braking(
                S.get('speedX', 0), S.get('angle', 0), self._calculate_curvature(S)
            )

        # Gear shifting
        R['gear'] = self._shift_gears(S)

        # Update state and reward
        reward = self.calculate_reward(S, R)
        self.update_models(S, R, reward)

        # Update previous values
        self.previous_steer = R['steer']
        self.previous_accel = R['accel']
        self.previous_brake = R['brake']

    def _shift_gears(self, S: Dict[str, Any]) -> int:
        """Optimized gear shifting."""
        speed = S.get('speedX', 0)
        if speed > 140:
            return 3
        elif speed > 100:
            return 2
        elif speed > 60:
            return 1
        return 0

    def calculate_reward(self, S: Dict[str, Any], R: Dict[str, float]) -> float:
        """Calculate reward for reinforcement learning."""
        reward = 0

        # Speed reward
        speed = S.get('speedX', 0)
        reward += speed * 0.01

        # Position reward
        track_pos = abs(S.get('trackPos', 0))
        if track_pos < 0.8:
            reward += 1.0
        elif track_pos < 1.0:
            reward += 0.5
        else:
            reward -= 2.0

        # Stability reward
        angle = abs(S.get('angle', 0))
        if angle < 0.2:
            reward += 0.5

        # Damage penalty
        damage = S.get('damage', 0)
        reward -= damage * 0.001

        return reward


class DataCollector:
    """Collects racing experiences for training."""

    def __init__(self):
        self.experiences: List[Dict[str, Any]] = []

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """Add a new experience."""
        self.experiences.append(experience)

        # Limit memory
        if len(self.experiences) > 50000:
            self.experiences = self.experiences[-50000:]