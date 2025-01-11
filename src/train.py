"""
The approach combines epsilon-greedy DQN with Replay Buffer and Target Network.

Code is partially inspired from the notebooks of the course (https://github.com/erachelson/RLclass_MVA)
"""

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

try:
    from fast_env import FastHIVPatient as HIVPatient
except ImportError:
    pass
from torch import nn
import random
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class DQNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, nb_neurons: int = 24):
        """DQN model"""
        super(DQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int, device):
        self.capacity = capacity  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        batch = random.sample(self.data, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch)))
        )

    def __len__(self):
        return len(self.data)


class ProjectAgent:
    def __init__(
        self,
        config={
            "nb_actions": env.action_space.n,
            "learning_rate": 0.0005,
            "gamma": 0.9,
            "buffer_size": 1000000,
            "epsilon_min": 0.10,
            "epsilon_max": 1.0,
            "epsilon_delay": 200 * 5,
            "epsilon_step": 0.01,
            "batch_size": 256,
            "update_target_strategy": "ema",
            "update_target_freq": 400,
            "update_target_tau": 0.005,
            "gradient_steps": 5,
            "state_dim": env.observation_space.shape[0],
        },
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        self.nb_actions = config["nb_actions"]
        self.state_dim = config["state_dim"]
        self.gamma = config["gamma"] if "gamma" in config.keys() else 0.95
        self.batch_size = config["batch_size"] if "batch_size" in config.keys() else 100
        buffer_size = (
            config["buffer_size"] if "buffer_size" in config.keys() else int(1e5)
        )
        self.memory = ReplayBuffer(capacity=buffer_size, device=self.device)
        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_delay = config["epsilon_delay"]
        self.epsilon_step = config["epsilon_step"]
        self.model = DQNetwork(self.state_dim, self.nb_actions).to(device)
        self.target_model = deepcopy(self.model).to(device)
        lr = config["learning_rate"] if "learning_rate" in config.keys() else 0.001
        self.optimizer = (
            config["optimizer"]
            if "optimizer" in config.keys()
            else torch.optim.Adam(self.model.parameters(), lr=lr)
        )
        self.nb_gradient_steps = config["gradient_steps"]
        self.update_target_strategy = (
            config["update_target_strategy"]
            if "update_target_strategy" in config.keys()
            else "replace"
        )
        self.update_target_freq = config["update_target_freq"]
        self.update_target_tau = (
            config["update_target_tau"]
            if "update_target_tau" in config.keys()
            else 0.005
        )
        self.scheduler = StepLR(self.optimizer, step_size=300, gamma=0.5)
        self.criterion = nn.MSELoss()

    def normalize_reward(self, reward: float, max_reward: float = 10000000.0) -> float:
        return reward / max_reward

    def update_epsilon(self, step: int, epsilon: float) -> float:
        if step < self.epsilon_delay:
            return self.epsilon_max
        else:
            return max(self.epsilon_min, epsilon - self.epsilon_step)

    def greedy_action(self, state_tensor):
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            Q_values = self.model(state_tensor.unsqueeze(0))
            action = torch.argmax(Q_values, dim=1).item()
        return action

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode: int):
        episode_return = []
        step, episode, episode_cum_reward, previous_episode_cum_reward = 0, 0, 0, 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        best_return = float("-inf")
        while episode < max_episode:

            state_tensor = torch.FloatTensor(state).to(self.device)

            # epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(state_tensor)

            # Step in the environment
            next_state, reward, done, trunc, _ = env.step(action)
            reward = self.normalize_reward(reward)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train the agent
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target network if needed
            if self.update_target_strategy == "replace":
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            elif self.update_target_strategy == "ema":
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = (
                        tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                    )
                self.target_model.load_state_dict(target_state_dict)

            # Next transition
            step += 1
            if done or trunc:
                episode += 1
                reward_formatted = "{:.2e}".format(episode_cum_reward * 10000000)
                print(
                    f"Episode {episode:4d}, epsilon {epsilon:6.2f}, batch size {len(self.memory):5d}, episode return {reward_formatted}"
                )

                if (
                    episode_cum_reward + previous_episode_cum_reward > best_return
                ):  # if score on random env + normal env is better than previous best
                    best_return = episode_cum_reward + previous_episode_cum_reward
                    self.save_model()

                if episode % 2 == 0:  # Randomize the environment every 2 episodes
                    env.domain_randomization = True
                else:
                    env.domain_randomization = False

                state, _ = env.reset()
                previous_episode_cum_reward = episode_cum_reward
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0

                # Update epsilon
                epsilon = self.update_epsilon(step=step, epsilon=epsilon)

                self.scheduler.step()
            else:
                state = next_state

        self.save_model("dqn_model_final.pth")
        return episode_return

    def save_model(self, path: str = "dqn_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")

    def load(self, path: str = "dqn_model_final.pth"):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

    def act(self, observation, use_random=False) -> int:
        state_tensor = torch.FloatTensor(observation).to(self.device)
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(state_tensor)


if __name__ == "__main__":
    agent = ProjectAgent()
    scores = agent.train(env, 400)
