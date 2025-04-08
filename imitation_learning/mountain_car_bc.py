import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cpu')


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='single_rgb_array')
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos

def pad_or_truncate(state, target_size=2):
    """
    Ensure the state has exactly `target_size` features by padding or truncating.
    """
    state = np.array(state, dtype=np.float32).flatten()
    if state.size < target_size:
        # Pad with zeros if the state has fewer features
        return np.pad(state, (0, target_size - state.size), 'constant')
    elif state.size > target_size:
        # Truncate if the state has more features
        return state[:target_size]
    return state


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


def train_policy(obs, acs, nn_policy, num_train_iters):
    """TODO: train the policy using standard behavior cloning. Feel free to add other helper methods if you'd like or restructure the code as desired."""

    optimizer = Adam(nn_policy.parameters(), lr=2e-1)  # Optimizer
    loss_fn = nn.CrossEntropyLoss()  # Loss function for classification

    for i in range(num_train_iters):
        optimizer.zero_grad()
        # Forward pass: Get action logits from the policy network
        logits = nn_policy(obs)

        # Compute the loss between predicted logits and actual actions
        loss = loss_fn(logits, acs)

        # Backward pass and optimization step
        
        loss.backward()
        optimizer.step()

        if i % 10 == 0:  # Print loss every 10 iterations for monitoring
            print(f"Iteration {i}/{num_train_iters}, Loss: {loss.item()}")

class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # First hidden layer: input_dim=2, output_dim=64
        self.fc3 = nn.Linear(8, 3)  # Output layer: logits for 3 actions


    def forward(self, x):
        """TODO: this method performs a forward pass through the network, applying a non-linearity (ReLU is fine) on the hidden layers and should output logit values (since this is a discrete action task) for the 3-way classification problem"""
        x = F.relu(self.fc1(x))  # Apply ReLU non-linearity to the first layer
        x = self.fc3(x)  # Output logits for 3 actions
        return x
    

#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0", render_mode='human')
    else:
        env = gym.make("MountainCar-v0")

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, done, info = env.step(action)
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=20, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()

    #collect human demos
    demos = collect_human_demos(args.num_demos)
    for i, demo in enumerate(demos):
        print(f"Demo {i}: {demo}")  # Debug structure of demos

    #process demos
    obs, acs, _ = torchify_demos(demos)

    #train policy
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)
