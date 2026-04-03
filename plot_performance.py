import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_and_save(log_pattern, output_path):
    log_files = sorted(glob.glob(log_pattern))
    print(f"Found {len(log_files)} seed files in {log_pattern}")

    all_rewards = []
    for f in log_files:
        df = pd.read_csv(f)
        all_rewards.append(df["total_reward"].values)

    min_len = min(len(r) for r in all_rewards)
    rewards = np.stack([r[:min_len] for r in all_rewards])

    episodes = np.arange(min_len)
    mean = rewards.mean(axis=0)
    sem = stats.sem(rewards, axis=0)
    ci95 = sem * stats.t.ppf(0.975, df=len(log_files) - 1)

    window = max(1, min_len // 50)
    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    ep_s = smooth(episodes, window) + window // 2
    mean_s = smooth(mean, window)
    ci95_s = smooth(ci95, window)

    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep_s, mean_s, label="Mean reward", color="steelblue", linewidth=1.8)
    ax.fill_between(ep_s, mean_s - ci95_s, mean_s + ci95_s, alpha=0.3, color="steelblue", label="95% CI")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close()


plot_and_save("logs-per/train_log_seed*.csv", "per_performance_plot.png")
