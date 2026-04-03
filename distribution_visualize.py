import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Replay factors used
replay_factors = [1, 2, 4, 8]
num_seeds = 15

# Store distributions
agg_perf = {}   # stores 15 values per replay factor

for rho in replay_factors:
    agg_perf[rho] = []
    for seed in range(num_seeds):
        file_path = f"logs-replay{rho}/train_log_seed{seed}.csv"

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Each episode reward = one sample
        rewards = df["total_reward"].values
        N=100

        # Aggregate across episodes (as required)

        aggregate_value = np.mean(rewards[-N:])

        agg_perf[rho].append(aggregate_value)


# Plotting
plt.figure(figsize=(8, 6))

for rho in replay_factors:
    data = agg_perf[rho]

    # KDE curve
    sns.kdeplot(data, fill=True, alpha=0.2, label=f"ρ = {rho}")

    # Optional: histogram (can comment out if cluttered)
    #plt.hist(data, bins=8, alpha=0.2, density=True)

plt.title("Performance Distribution Across Replay Factors")
plt.xlabel("Aggregate Performance (Mean of Last 100 Episodes)")
plt.ylabel("Density")

plt.legend()
plt.grid()

plt.savefig("combined_performance_distribution.png", dpi=300, bbox_inches='tight')
plt.show()