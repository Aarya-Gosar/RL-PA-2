import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

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


checkpoint_path = "checkpoints/checkpoint_seed13_ep1999.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

model = DQN(n_observations=2, n_actions=3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=2000)
state, _ = env.reset()
total_reward = 0
done = False

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state_tensor).argmax().item()
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished — total reward: {total_reward}")
env.close()
