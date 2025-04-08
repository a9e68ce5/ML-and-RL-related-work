import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """Build a feedforward neural network."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], 
          lr=1e-2, epochs=50, batch_size=5000, render=False):
    """
    Train a policy network and return a list of average returns per epoch.
    After each epoch, a rollout is rendered from a second environment (with render_mode="human")
    to visualize the current policy's performance.
    """
    # Create environment for training
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for environments with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for environments with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Build policy network
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # Training-time get_action (gradients are tracked)
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    # Evaluation-time get_action (using no_grad for test-time forward pass)
    def get_action_eval(obs):
        with torch.no_grad():
            return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    # To track average returns at each epoch
    avg_returns_each_epoch = []

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, info = env.reset()
        done = False
        ep_rews = []

        # No visual rendering during training collection
        while True:
            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                # Use episode return as the weight for each state in the episode
                batch_weights += [ep_ret] * ep_len

                obs, info = env.reset()
                done, ep_rews = False, []

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)

    def render_rollout():
        """
        Create a new environment with render_mode="human" and run one episode
        using the current policy. Use get_action_eval so that no gradients are computed.
        """
        # Create a second environment for rendering
        env_render = gym.make(env_name, render_mode="human")
        obs, info = env_render.reset()
        done = False
        total_reward = 0
        while not done:
            # For environments with render_mode="human", env_render.render() may be optional,
            # but is called here for clarity.
            env_render.render()
            act = get_action_eval(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env_render.step(act)
            done = terminated or truncated
            total_reward += rew
        print(f"Rendered rollout total reward: {total_reward}")
        env_render.close()

    # Main training loop
    for i in range(epochs):
        batch_loss, mean_ret, mean_len = train_one_epoch()
        avg_returns_each_epoch.append(mean_ret)
        print(f"epoch: {i:3d}\t loss: {batch_loss:.3f}\t return: {mean_ret:.3f}\t ep_len: {mean_len:.3f}")

        # After each epoch, visually render one rollout to see the current policy in action
        render_rollout()

    env.close()
    return avg_returns_each_epoch

if __name__ == '__main__':
    # 1) Define the hyper-parameters to test
    lrs = [1e-2, 1e-3]     # different learning rates
    epochs_list = [20, 50] # different numbers of epochs

    # 2) Train for each (lr, epochs) pair and store results
    results = {}  # dict to store { (lr, epochs): [list of returns per epoch], ... }

    for lr in lrs:
        for ep in epochs_list:
            print("\n======================================")
            print(f"Start training: lr={lr}, epochs={ep}")
            avg_returns = train(env_name='CartPole-v1', 
                                hidden_sizes=[32], 
                                lr=lr, 
                                epochs=ep, 
                                batch_size=5000, 
                                render=False)
            results[(lr, ep)] = avg_returns

    # 3) Plot each result
    for key, returns in results.items():
        lr_val, ep_val = key
        plt.figure()
        plt.plot(range(len(returns)), returns)
        plt.title(f"Training Curve (lr={lr_val}, epochs={ep_val})")
        plt.xlabel("Epoch")
        plt.ylabel("Average Return")
        plt.show()
