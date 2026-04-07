import gymnasium as gym
import matplotlib.pyplot as plt
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
import numpy as np
class DQN(nn.Module):
    def __init__(self, n_observations  = 2, n_actions = 3):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32,32)
        self.layer3 = nn.Linear(32,n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def sample(buffer , batch_size):
    combined_batch = []
    for action in range(3):
        if(len(buffer[action]) > batch_size):
            combined_batch.append(random.sample(buffer[action],batch_size))
        else:
            return []
        
    return combined_batch



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

    #forward pass gives value for each action, we need only for the action we picked.
    Q = policy_network(states).gather(1, actions).squeeze()
    with torch.no_grad():
        next_Q = target_network(next_states).max(1)[0]
        #Only adds reward if next state is terminal
        target_Q = rewards + (1 - terminals) * gamma * next_Q
    loss = F.mse_loss(Q, target_Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def hard_update(policy_network,target_network):
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
    
    # Also save a 'latest' version for easy access
    latest_path = os.path.join(folder, f"latest_seed{seed}.pth")
    torch.save(state_dict, latest_path)
    print(f"--> Saved checkpoint: {checkpoint_path}")


def log_to_csv(episode, reward, epsilon, steps, seed, folder="logs"):
    """Appends episode data to a CSV file."""
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
memory_size =  min(num_episodes,2000) * truncation_length #per action | roughly 50 episodes
ideal_batch_size = 32 # For each action
ideal_hard_update_time = 10000 #roughly 5 episodes

hyper_params_weights = [(1/4,1) , (1/2 , 1) , (2,1) , (4,1) , (1,1/25) , (1,1/5) , (1,5) , (1,25)]


gamma = 0.99
epsilon = 0.99
alpha = 0.1
env = gym.make("MountainCar-v0" , max_episode_steps=truncation_length)
# Observation space = [position,velocity]
# Action space  = [0,1,2] left,stay,right
env.reset()

rewards_all_seeds = []
num_seeds = 15

truncation_length = 2000
replay_factors_list = [1,4]

for seed in range(num_seeds):
    env = gym.make("MountainCar-v0" , max_episode_steps=truncation_length)
    env.reset()

    for replay_factor in replay_factors_list:
        for weights in hyper_params_weights:
            batch_size = int(ideal_batch_size * weights[0])
            hard_update_time = int(ideal_hard_update_time * weights[1])
            print(f"SEED Number: {seed} | weights: {weights} | Rho: {replay_factor}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
            env = gym.make("MountainCar-v0", max_episode_steps=truncation_length)


            #buffer[a] = memory for action = a 
            #if maxlen is reached  then deque will pop first element while appending
            #Stores: [state,action,reward,next_state , terminal] 
            epsilon = 0.99
            buffer = {a : deque(maxlen=memory_size) for a in range(3)}
            policy_network = DQN(n_observations=2, n_actions=3)
            target_network = DQN(n_observations=2, n_actions=3)
            target_network.load_state_dict(policy_network.state_dict())
            optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)



            global_steps = 0
            rewards = []
            for episode in tqdm(range(num_episodes)):
                
                state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
                prev_state = state
                total_reward = 0
                for t in range(truncation_length):
                    global_steps += 1
                    state = prev_state
                    if(random.random() < epsilon):
                        action = env.action_space.sample()
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        action = policy_network(state_tensor).argmax().item()  

                    next_state , reward , terminated , truncated , _ = env.step(action)
                    total_reward += reward
                    terminal = terminated or truncated
                    buffer[action].append([state,action,reward,next_state,terminal])
                    prev_state = next_state
                    for _ in range(replay_factor):
                        improve_policy(buffer, policy_network, target_network, optimizer, batch_size, gamma)
                    if(global_steps % hard_update_time == 0):
                        hard_update(policy_network,target_network)
                    if(terminal):
                        break
                rewards.append(total_reward)


                #Log our data
                folder_suffix = f"rf{replay_factor}_w{weights[0]}-{weights[1]}"
                log_folder = f"logs-sens-{folder_suffix}"
                ckpt_folder = f"checkpoints-sens-{folder_suffix}"
                log_to_csv(episode, total_reward, epsilon, t, seed,folder=log_folder)
                if episode % 50 == 0 or episode == num_episodes - 1:
                    save_checkpoint(
                        policy_network.state_dict(), 
                        optimizer.state_dict(), 
                        episode, 
                        epsilon, 
                        seed,
                        folder = ckpt_folder
                    )
                epsilon = max(0.01, epsilon * 0.99)

