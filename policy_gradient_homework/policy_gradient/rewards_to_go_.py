import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt

# -----------------------------------------
# 1) MLP and utility
# -----------------------------------------
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    """
    Given a list of rewards for a single episode, compute 'reward-to-go' by summing
    future rewards from each index.
    """
    n = len(rews)
    rtg = np.zeros_like(rews, dtype=np.float32)
    running_sum = 0.0
    for i in reversed(range(n)):
        running_sum += rews[i]
        rtg[i] = running_sum
    return rtg

# -----------------------------------------
# 2) Vanilla Policy Gradient (no reward-to-go)
# -----------------------------------------
def train_vanilla(env_name='CartPole-v1', hidden_sizes=[32],
                  lr=1e-2, epochs=50, batch_size=5000, seed=0):
    """
    'Vanilla' policy gradient: we weight each timestep by the total episode return.
    Returns a list of average returns (mean episode return each epoch).
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Build policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)
    avg_returns_each_epoch = []

    for epoch in range(epochs):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, info = env.reset()
        done = False
        ep_rews = []

        while True:
            batch_obs.append(obs.copy())
            act = get_policy(torch.as_tensor(obs, dtype=torch.float32)).sample().item()
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # For vanilla PG, every state in this episode gets the total return
                batch_weights += [ep_ret] * ep_len

                obs, info = env.reset()
                done, ep_rews = False, []

                if len(batch_obs) > batch_size:
                    break

        # Update network
        optimizer.zero_grad()
        loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                            act=torch.as_tensor(batch_acts, dtype=torch.int32),
                            weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        avg_returns_each_epoch.append(np.mean(batch_rets))

    env.close()
    return avg_returns_each_epoch


# -----------------------------------------
# 3) Reward-to-Go Policy Gradient
# -----------------------------------------
def train_rtg(env_name='CartPole-v1', hidden_sizes=[32],
              lr=1e-2, epochs=50, batch_size=5000, seed=0):
    """
    Reward-to-Go policy gradient: we weight each timestep by the sum of rewards
    from t onward in the episode.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)
    avg_returns_each_epoch = []

    for epoch in range(epochs):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, info = env.reset()
        done = False
        ep_rews = []

        while True:
            batch_obs.append(obs.copy())
            act = get_policy(torch.as_tensor(obs, dtype=torch.float32)).sample().item()
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # For RTG, each state in this episode gets the sum of rewards from that step onward
                rtg = reward_to_go(ep_rews)
                batch_weights += list(rtg)

                obs, info = env.reset()
                done, ep_rews = False, []

                if len(batch_obs) > batch_size:
                    break

        # Update network
        optimizer.zero_grad()
        loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                            act=torch.as_tensor(batch_acts, dtype=torch.int32),
                            weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        avg_returns_each_epoch.append(np.mean(batch_rets))

    env.close()
    return avg_returns_each_epoch


# -----------------------------------------
# 4) Compare the two methods
# -----------------------------------------
if __name__ == '__main__':

    import statistics

    # Hyperparameters
    lr = 1e-2
    epochs = 50
    n_runs = 3  # number of runs per method

    # Store results for each run
    vanilla_runs = []
    rtg_runs = []

    for seed in range(n_runs):
        print(f"\n=== Vanilla PG: Run {seed+1} / {n_runs} ===")
        returns_v = train_vanilla(env_name='CartPole-v1',
                                  hidden_sizes=[32],
                                  lr=lr,
                                  epochs=epochs,
                                  batch_size=5000,
                                  seed=seed)
        vanilla_runs.append(returns_v)

    for seed in range(n_runs):
        print(f"\n=== RTG PG: Run {seed+1} / {n_runs} ===")
        returns_rtg = train_rtg(env_name='CartPole-v1',
                                hidden_sizes=[32],
                                lr=lr,
                                epochs=epochs,
                                batch_size=5000,
                                seed=seed)
        rtg_runs.append(returns_rtg)

    # Convert to NumPy for easier manipulation
    vanilla_runs = np.array(vanilla_runs)  # shape = [n_runs, epochs]
    rtg_runs = np.array(rtg_runs)          # shape = [n_runs, epochs]

    # Compute mean and std across runs
    vanilla_mean = np.mean(vanilla_runs, axis=0)
    vanilla_std  = np.std(vanilla_runs, axis=0)

    rtg_mean = np.mean(rtg_runs, axis=0)
    rtg_std  = np.std(rtg_runs, axis=0)

    # -----------------------------------------
    # Plotting
    # -----------------------------------------
    epochs_x = np.arange(epochs)

    plt.figure()
    # Plot vanilla
    plt.plot(epochs_x, vanilla_mean, label='Vanilla PG (mean)')
    plt.fill_between(epochs_x,
                     vanilla_mean - vanilla_std,
                     vanilla_mean + vanilla_std,
                     alpha=0.2)

    # Plot RTG
    plt.plot(epochs_x, rtg_mean, label='RTG PG (mean)')
    plt.fill_between(epochs_x,
                     rtg_mean - rtg_std,
                     rtg_mean + rtg_std,
                     alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title("Vanilla vs. Reward-to-Go Policy Gradient")
    plt.legend()
    plt.show()
