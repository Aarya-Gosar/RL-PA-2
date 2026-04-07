import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, nct

replay_factors = [1, 2, 4, 8]
num_seeds = 15
ALPHA = 0.05
BETA = 0.90
SMOOTH_W = 50  # rolling window for smoothing

def tolerance_factor(n, alpha=0.05, beta=0.90):
    z_beta = norm.ppf((1 + beta) / 2)
    k = nct.ppf(1 - alpha / 2, df=n - 1, nc=np.sqrt(n) * z_beta) / np.sqrt(n)
    return k

def smooth(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values



# Read logs
returns_over_time = {}
for rho in replay_factors:
    returns_over_time[rho] = []
    for seed in range(num_seeds):
        file_path = f"logs-replay{rho}/train_log_seed{seed}.csv"
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        returns_over_time[rho].append(df["total_reward"].values)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

for rho in replay_factors:
    series = returns_over_time[rho]
    if not series:
        continue

    min_len = min(len(s) for s in series)
    # Smooth each seed first, then compute stats across seeds
    mat = np.array([smooth(s[:min_len], SMOOTH_W) for s in series])
    n = mat.shape[0]
    k = tolerance_factor(n, ALPHA, BETA)

    mean = mat.mean(axis=0)
    std  = mat.std(axis=0, ddof=1)

    ax.plot(mean,  linewidth=1.8, label=f"ρ = {rho}")
    ax.fill_between(np.arange(min_len), mean - k * std, mean + k * std,
                     alpha=0.2)

ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title(f"Mean performance with (α={ALPHA}, β={BETA}) tolerance intervals")
ax.legend()
plt.tight_layout()
plt.savefig("tolerance_intervals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Done.")