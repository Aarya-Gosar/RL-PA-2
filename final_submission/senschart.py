import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings

# Ignore specific RuntimeWarnings globally
warnings.filterwarnings('ignore')

def calculate_95_ci(data):
    """Calculates the mean and 95% confidence interval for a given array of data."""
    n = len(data)
    if n < 2:
        return np.mean(data) if n == 1 else 0.0, 0.0
    mean = np.mean(data)
    std_err = st.sem(data)
    ci = std_err * st.t.ppf((1 + 0.95) / 2., n - 1)
    return mean, ci

def get_auc_stats(folder_path, num_seeds=15):
    """Reads logs from a folder and returns the normalized mean AUC and 95% CI."""
    aucs = []
    for seed in range(num_seeds):
        log_path = os.path.join(folder_path, f"train_log_seed{seed}.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            rewards = df['total_reward'].values
            
            # Calculate Area Under the Curve and DIVIDE BY 2000
            auc = np.trapz(rewards) / 2000.0
            aucs.append(auc)
            
    if not aucs:
        return np.nan, np.nan
        
    return calculate_95_ci(aucs)

def main():
    replay_factors = [1, 4]
    num_seeds = 15
    
    # Baseline hyperparameters
    ideal_batch_size = 32
    ideal_update_time = 10000
    
    # Python evaluates 1/4 to 0.25, 1/25 to 0.04, etc. 
    batch_weights = [(0.25, 1), (0.5, 1), (1, 1), (2, 1), (4, 1)]
    update_weights = [(1, 0.04), (1, 0.2), (1, 1), (1, 5), (1, 25)]
    
    # Map to absolute values for the X-axis
    batch_x_vals = [ideal_batch_size * w[0] for w in batch_weights]
    update_x_vals = [ideal_update_time * w[1] for w in update_weights]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {1: '#1f77b4', 4: '#2ca02c'} # Blue and Green
    
    # ---------------------------------------------------------
    # Plot 1 & Printout: Sensitivity to Mini-Batch Size
    # ---------------------------------------------------------
    print("-" * 65)
    print("SENSITIVITY TO MINI-BATCH SIZE (Normalized AUC)")
    print("-" * 65)
    print(f"{'Batch Size':<15} | {'Rho':<5} | {'Mean AUC':<15} | {'95% CI (+/-)'}")
    print("-" * 65)
    
    ax1 = axes[0]
    for rho in replay_factors:
        means, cis = [], []
        for i, w in enumerate(batch_weights):
            if w == (1, 1):
                folder = f"logs-replay{rho}"
            else:
                folder = f"logs-sens-rf{rho}_w{w[0]}-{w[1]}" 
                
            mean, ci = get_auc_stats(folder, num_seeds)
            means.append(mean)
            cis.append(ci)
            
            # Print the extracted values to the console
            batch_val = batch_x_vals[i]
            print(f"{batch_val:<15.1f} | {rho:<5} | {mean:<15.4f} | {ci:.4f}")
            
        ax1.errorbar(batch_x_vals, means, yerr=cis, fmt='-o', capsize=5, 
                     color=colors[rho], label=f'Replay factor $\\rho$={rho}', markersize=6)

    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batch_x_vals)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
    ax1.set_xlabel('Mini-batch Size')
    ax1.set_ylabel('Normalized Performance (AUC / 2000)')
    ax1.set_title('(a) Sensitivity to Mini-batch Size')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # ---------------------------------------------------------
    # Plot 2 & Printout: Sensitivity to Target Network Refresh Rate
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print("SENSITIVITY TO TARGET REFRESH RATE (Normalized AUC)")
    print("-" * 65)
    print(f"{'Refresh Rate':<15} | {'Rho':<5} | {'Mean AUC':<15} | {'95% CI (+/-)'}")
    print("-" * 65)
    
    ax2 = axes[1]
    for rho in replay_factors:
        means, cis = [], []
        for i, w in enumerate(update_weights):
            if w == (1, 1):
                folder = f"logs-replay{rho}"
            else:
                folder = f"logs-sens-rf{rho}_w{w[0]}-{w[1]}"
                
            mean, ci = get_auc_stats(folder, num_seeds)
            means.append(mean)
            cis.append(ci)
            
            # Print the extracted values to the console
            refresh_val = update_x_vals[i]
            print(f"{refresh_val:<15.1f} | {rho:<5} | {mean:<15.4f} | {ci:.4f}")
            
        ax2.errorbar(update_x_vals, means, yerr=cis, fmt='-o', capsize=5, 
                     color=colors[rho], label=f'Replay factor $\\rho$={rho}', markersize=6)

    ax2.set_xscale('log', base=10)
    ax2.set_xticks(update_x_vals)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
    ax2.set_xlabel('Target Network Refresh Rate (steps)')
    ax2.set_title('(b) Sensitivity to Target Network Refresh Rate')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('sensitivity_analysis_curves.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()