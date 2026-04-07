import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np
import os
import csv

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PERBuffer:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, transition):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # fallback: resample from full range
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


class DQN(nn.Module):
    def __init__(self, n_observations=2, n_actions=3):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def improve_policy(buffer, policy_network, target_network, optimizer, batch_size, gamma, beta):
    if len(buffer) < batch_size * 3:
        return

    batch, indices, weights = buffer.sample(batch_size * 3, beta)

    batch_array = np.array(batch, dtype=object)
    states = torch.tensor(np.stack(batch_array[:, 0]), dtype=torch.float32)
    actions = torch.tensor(batch_array[:, 1].astype(int), dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(batch_array[:, 2].astype(float), dtype=torch.float32)
    next_states = torch.tensor(np.stack(batch_array[:, 3]), dtype=torch.float32)
    terminals = torch.tensor(batch_array[:, 4].astype(float), dtype=torch.float32)

    Q = policy_network(states).gather(1, actions).squeeze()
    with torch.no_grad():
        next_Q = target_network(next_states).max(1)[0]
        target_Q = rewards + (1 - terminals) * gamma * next_Q

    td_errors = (Q - target_Q).detach().cpu().numpy()
    buffer.update_priorities(indices, td_errors)

    loss = (weights * F.mse_loss(Q, target_Q, reduction='none')).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def hard_update(policy_network, target_network):
    target_network.load_state_dict(policy_network.state_dict())


def save_checkpoint(state_dict, optimizer_dict, episode, epsilon, seed, folder="checkpoints"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    checkpoint_path = os.path.join(folder, f"checkpoint_seed{seed}_ep{episode}.pth")

    torch.save({
        'episode': episode,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer_dict,
        'epsilon': epsilon,
    }, checkpoint_path)

    latest_path = os.path.join(folder, f"latest_seed{seed}.pth")
    torch.save(state_dict, latest_path)
    print(f"--> Saved checkpoint: {checkpoint_path}")


def log_to_csv(episode, reward, epsilon, steps, seed, folder="logs"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    log_path = os.path.join(folder, f"train_log_seed{seed}.csv")
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode', 'total_reward', 'epsilon', 'total_steps'])
        writer.writerow([episode, reward, epsilon, steps])


truncation_length = 2000
num_episodes = 2000
memory_size = min(2000, num_episodes) * truncation_length
batch_size = 32
hard_update_time = 10000
replay_factor = 4

gamma = 0.99
epsilon = 0.99
per_alpha = 0.6
per_beta_start = 0.4
per_beta_end = 1.0

rewards_all_seeds = []
num_seeds = 15

trunc_lens = [2000]

print(torch.cuda.is_available())          # True = GPU detected
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
t = torch.FloatTensor([1.0]).to(device)
print(t.device)  

for truncation_length in trunc_lens:
    env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)
    env.reset()

    for seed in range(num_seeds):
        print(f"SEED Number: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)

        epsilon = 0.99
        buffer = PERBuffer(capacity=memory_size, alpha=per_alpha)
        policy_network = DQN(n_observations=2, n_actions=3)
        target_network = DQN(n_observations=2, n_actions=3)
        target_network.load_state_dict(policy_network.state_dict())
        optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)

        global_steps = 0
        rewards = []
        for episode in tqdm(range(num_episodes)):
            # Anneal beta from per_beta_start to per_beta_end over training
            beta = per_beta_start + (per_beta_end - per_beta_start) * (episode / num_episodes)

            state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
            prev_state = state
            total_reward = 0
            for t in range(truncation_length):
                global_steps += 1
                state = prev_state
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = policy_network(state_tensor).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                terminal = terminated or truncated
                buffer.add([state, action, reward, next_state, terminal])
                prev_state = next_state
                for _ in range(replay_factor):
                    improve_policy(buffer, policy_network, target_network, optimizer, batch_size, gamma, beta)
                if global_steps % hard_update_time == 0:
                    hard_update(policy_network, target_network)
                if terminal:
                    break
            rewards.append(total_reward)

            log_to_csv(episode, total_reward, epsilon, t, seed, folder=f"logs-per-replay{replay_factor}")
            if episode % 50 == 0 or episode == num_episodes - 1:
                save_checkpoint(
                    policy_network.state_dict(),
                    optimizer.state_dict(),
                    episode,
                    epsilon,
                    seed,
                    folder=f"checkpoints-per-replay{replay_factor}"
                )

            epsilon = max(0.01, epsilon * 0.99)

        rewards_all_seeds.append(rewards)
