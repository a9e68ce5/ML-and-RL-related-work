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
    """
    # Create the main (training) environment
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for environments with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for environments with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Build the policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        """ Sample an action from the policy (used during training). """
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    # Track average returns at each epoch
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

        while True:
            # Collect data for training
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
                # Use the episode return as the weight for each state in the episode
                batch_weights += [ep_ret] * ep_len

                obs, info = env.reset()
                done, ep_rews = False, []

                # End the loop if we have enough data
                if len(batch_obs) > batch_size:
                    break

        # Take a single policy gradient update
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)

    # --- Additional environment for rendering a test episode each epoch ---
    # Only create this if we actually want to render
    if render:
        env_render = gym.make(env_name, render_mode="human")
    else:
        env_render = None

    def watch_agent():
        """
        Run exactly one episode in the render-mode environment, 
        using the current policy with torch.no_grad() so we don't track gradients.
        """
        if env_render is None:
            return

        obs, info = env_render.reset()
        done = False
        while not done:
            # Don't track gradients while testing/watching
            with torch.no_grad():
                action = get_policy(torch.as_tensor(obs, dtype=torch.float32)).sample().item()

            obs, rew, terminated, truncated, info = env_render.step(action)
            done = terminated or truncated
        # After the episode ends, it will automatically reset next time we call watch_agent()

    # Main training loop
    for i in range(epochs):
        batch_loss, mean_ret, mean_len = train_one_epoch()
        avg_returns_each_epoch.append(mean_ret)

        print(f"epoch: {i:3d}\t loss: {batch_loss:.3f}\t"
              f" return: {mean_ret:.3f}\t ep_len: {mean_len:.3f}")

        # Render one rollout from the current policy
        watch_agent()

    # Close environments
    env.close()
    if env_render is not None:
        env_render.close()

    return avg_returns_each_epoch

if __name__ == '__main__':
    # Example usage: call train with render=True to visually see the agent each epoch
    train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-3, epochs=10, batch_size=2000, render=True)
