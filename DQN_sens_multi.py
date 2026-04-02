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

# --- Core Model ---
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

# --- Helper Functions ---
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

def log_to_csv(episode, reward, epsilon, steps, seed, folder):
    os.makedirs(folder, exist_ok=True)
    log_path = os.path.join(folder, f"train_log_seed{seed}.csv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode', 'total_reward', 'epsilon', 'total_steps'])
        writer.writerow([episode, reward, epsilon, steps])

def save_checkpoint(state_dict, optimizer_dict, episode, epsilon, seed, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"checkpoint_seed{seed}_ep{episode}.pth")
    torch.save({'model_state_dict': state_dict, 'epsilon': epsilon}, path)

# --- The Parallel Worker ---
def train_config(params):
    seed, replay_factor, weights, config_id = params
    
    # Unpack hyperparams
    ideal_batch_size = 32
    ideal_hard_update_time = 10000
    batch_size = int(ideal_batch_size * weights[0])
    hard_update_time = int(ideal_hard_update_time * weights[1])
    num_episodes = 2000
    truncation_length = 2000
    memory_size = min(num_episodes,2000) * truncation_length 
    gamma = 0.99
    
    # Local process seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)
    
    policy_network = DQN()
    target_network = DQN()
    target_network.load_state_dict(policy_network.state_dict())
    optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)
    buffer = {a: deque(maxlen=memory_size) for a in range(3)}
    
    epsilon = 0.99
    global_steps = 0
    
    # Unique folders for this specific weight/factor combination
    folder_suffix = f"rf{replay_factor}_w{weights[0]}-{weights[1]}"
    log_folder = f"logs-sens-{folder_suffix}"
    ckpt_folder = f"checkpoints-sens-{folder_suffix}"

    try:
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
                
                if terminal: break

            log_to_csv(episode, total_reward, epsilon, t, seed, log_folder)
            if episode % 500 == 0: # Checkpoint less frequently in parallel to save IO
                save_checkpoint(policy_network.state_dict(), optimizer.state_dict(), episode, epsilon, seed, ckpt_folder)
            
            epsilon = max(0.01, epsilon * 0.99)
            
    except Exception as e:
        return f"Error in Task {config_id}: {str(e)}"
    
    return f"Success: Seed {seed}, weights {weights}"

# --- Execution Entry Point ---
if __name__ == "__main__":
    num_seeds = 4
    start_seed = 8
    replay_factors_list = [4]
    hyper_params_weights = [(1/4,1), (1/2,1), (2,1), (4,1), (1,1/25), (1,1/5), (1,5), (1,25)]
    
    # Flatten all tasks into a single list
    tasks = []
    task_counter = 0
    for seed in range(start_seed,start_seed+num_seeds):
        for rf in replay_factors_list:
            for weights in hyper_params_weights:
                tasks.append((seed, rf, weights, task_counter))
                task_counter += 1

    print(f"Total tasks to run: {len(tasks)}")
    
    # Use ProcessPoolExecutor. 
    # Adjust max_workers based on your RAM (MountainCar is light, but deques take space)
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
        # tqdm shows progress as tasks finish
        results = list(tqdm(executor.map(train_config, tasks), total=len(tasks)))

    for r in results[:5]: # Print first few results
        print(r)