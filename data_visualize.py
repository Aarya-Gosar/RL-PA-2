import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Define your folders and a list of colors for the plot
folders = ["logs-replay1", "logs-replay2", "logs-replay4", "logs-replay8"]
colors = ['blue', 'red', 'green' , "orange"]
window_size = 50 

def moving_avg(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def process_folder(folder_path):
    all_rewards = []
    # Loop through available seeds in the folder
    for i in range(0,15):
        file_path = f"{folder_path}/train_log_seed{i}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_rewards.append(df['total_reward'].values)
    
    # Standardize length in case of minor discrepancies
    min_len = min(len(r) for r in all_rewards)
    rewards_array = np.array([r[:min_len] for r in all_rewards])
    
    mean = np.mean(rewards_array, axis=0)
    std = np.std(rewards_array, axis=0)
    return mean, std, min_len

plt.figure(figsize=(12, 7))

for folder, color in zip(folders, colors):
    if not os.path.exists(folder):
        print(f"Warning: {folder} not found. Skipping...")
        continue
        
    mean_rewards, std_rewards, length = process_folder(folder)
    
    # Calculate bounds
    upper_bound = mean_rewards + std_rewards
    lower_bound = mean_rewards - std_rewards
    
    # Apply Smoothing
    smooth_mean = moving_avg(mean_rewards, window_size)
    smooth_upper = moving_avg(upper_bound, window_size)
    smooth_lower = moving_avg(lower_bound, window_size)
    
    # Adjust x-axis for 'valid' convolution
    x_axis = np.arange(window_size - 1, length)
    
    # Plot Mean
    plt.plot(x_axis, smooth_mean, label=f'Mean: {folder}', color=color, linewidth=2)
    
    # Plot Shaded Area
    plt.fill_between(x_axis, smooth_lower, smooth_upper, color=color, alpha=0.15)

# Final Plot Styling
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.title("DQN Comparative Performance Across Different Runs", fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()