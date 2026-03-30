import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from tqdm import tqdm
import numpy as np
import os
import csv
from concurrent.futures import ProcessPoolExecutor

# --- Model and Helper Classes ---
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

def improve_policy(buffer, policy_network, target_network, optimizer, batch_size, gamma):    
    if any(len(buffer[a]) < batch_size for a in range(3)):
        return

    batch = []
    for action in range(3):
        batch.extend(random.sample(buffer[action], batch_size))
    random.shuffle(batch)

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
    
    loss = F.mse_loss(Q, target_Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_checkpoint(state_dict, optimizer_dict, episode, epsilon, seed, folder="checkpoints"):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    checkpoint_path = os.path.join(folder, f"checkpoint_seed{seed}_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer_dict,
        'epsilon': epsilon,
    }, checkpoint_path)

def log_to_csv(episode, reward, epsilon, steps, seed, folder="logs"):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_path = os.path.join(folder, f"train_log_seed{seed}.csv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode', 'total_reward', 'epsilon', 'total_steps'])
        writer.writerow([episode, reward, epsilon, steps])

def train_seed(params):
    seed, replay_factor, num_episodes, truncation_length, memory_size, batch_size, hard_update_time, gamma = params
    
    print(f"Starting Seed: {seed} | Replay Factor: {replay_factor}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)
    
    buffer = {a: deque(maxlen=memory_size) for a in range(3)}
    policy_network = DQN(n_observations=2, n_actions=3)
    target_network = DQN(n_observations=2, n_actions=3)
    target_network.load_state_dict(policy_network.state_dict())
    optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)

    epsilon = 0.99
    global_steps = 0
    rewards = []

    log_folder = f"logs-replay{replay_factor}"
    ckpt_folder = f"checkpoints-replay{replay_factor}"

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
        total_reward = 0
        
        for t in range(truncation_length):
            global_steps += 1
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = policy_network(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            terminal = terminated or truncated
            buffer[action].append([state, action, reward, next_state, terminal])
            state = next_state

            for _ in range(replay_factor):
                improve_policy(buffer, policy_network, target_network, optimizer, batch_size, gamma)

            if global_steps % hard_update_time == 0:
                target_network.load_state_dict(policy_network.state_dict())

            if terminal:
                break
        
        rewards.append(total_reward)
        log_to_csv(episode, total_reward, epsilon, t, seed, folder=log_folder)
        
        if episode % 100 == 0:
            save_checkpoint(policy_network.state_dict(), optimizer.state_dict(), episode, epsilon, seed, folder=ckpt_folder)

        epsilon = max(0.01, epsilon * 0.99)
    
    return rewards

if __name__ == "__main__":
    truncation_length = 2000
    num_episodes = 2000  
    memory_size = min(num_episodes,2000) * truncation_length
    batch_size = 32
    hard_update_time = 10000
    gamma = 0.99
    replay_factors_list = [2,4]
    num_seeds = 15
    start_seed = 5
    
    tasks = []
    for rf in replay_factors_list:
        for s in range(start_seed, num_seeds):
            tasks.append((s, rf, num_episodes, truncation_length, memory_size, batch_size, hard_update_time, gamma))

    print(f"Launching {len(tasks)} parallel jobs...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(train_seed, tasks), total=len(tasks)))

    print("All training runs completed.")