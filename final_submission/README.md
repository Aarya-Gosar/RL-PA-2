# Requirements 

```
gymnasium==1.2.3
matplotlib==3.10.8
numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
seaborn==0.13.2
torch==2.5.1+cu121
tqdm==4.67.1


python version: 3.12.5
```
```bash
pip install -r requirements.txt
```
# Mountain Car Base version
Runs the standard DQN agent across multiple truncation lengths and seeds.

```bash
python DQN_mountcar.py
```

Key variables to adjust include trunc_lens (list of truncation lengths to test) and num_seeds (number of seeds to run for statistical significance).

### Output
`/logs-{truncation_length}/` Contains the .csv logs of all seeds for a given truncation length (e.g., 2000, 1000, 200).

`checkpoints-{truncation_length}/`: Stores the PyTorch model weights (.pth). It saves a checkpoint every 50 episodes, as well as a continuously updated latest_seed{seed}.pth version.

# Replay factor
Tests the effect of performing multiple optimization steps (replay factor) per environment step.

```bash
python DQN_replayfactor.py
```
Key variables to adjust in the script include replay_factors_list (controls the parameter rho) and num_seeds.

### Output
`/logs-{replay_factor}` Contains the training logs for the given replay factors (e.g., 2, 4, 8).

`/checkpoints-{replay_factor}/` Stores the PyTorch model weights (.pth), saving every 50 episodes and maintaining a "latest" version


# Sensitivity
Evaluates the agent's sensitivity to batch size and target network hard update frequencies.

```bash
python DQN_sensitivity.py
```

`hyper_params_weights` contains list of weights `(w1,w2)`.

`batch_size = ideal_batch_size  * w1` 

`hard_update_time = ideal_hard_update_time * w2`

### Output
`/logs-sens-rf{replay_factor}_w{w1}-{w2}` folder has all the logs

`/checkpoints-sens-rf{replay_factor}_w{w1}-{w2}`  Stores the PyTorch model weights (.pth). It saves a checkpoint every 50 episodes, as well as a "latest" version.


# Prioritized Experienced replay
Implements Prioritized Experience Replay to sample more significant transitions more frequently during training.

```bash
python DQN_PER.py
```

### Output
`/logs-per-replay{replay_factor}/` Contains the training logs utilizing prioritized sampling.

`/checkpoints-per-replay{replay_factor}/` Stores the PyTorch model weights (.pth), saving every 50 episodes and maintaining a "latest" version.

## Analysis code

This script is used to plot the confidence intervals of our results. 

We have included logs to plot them for the base case with `replay_factor=1`

```bash
python confidence_plot.py
```


**_NOTE:_**: We have also included code for other statistical analysis but not the logs to run the same. They wont run without generating the logs

namely:
`tolerance_intervals.py , distribution_visualize.py , senschart.py`




