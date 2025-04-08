# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_bandits = 10  # Number of arms
n_tasks = 1500  # Number of bandit tasks (less than 2000 to speed up)
n_plays = 1000  # Number of plays per task
epsilons = [0, 0.01, 0.1]  # Epsilon values

# Generate the true action values for each task
np.random.seed(42)  # For reproducibility
true_action_values = np.random.normal(0, 1, (n_tasks, n_bandits))

# Initialize storage for average rewards and optimal action percentage
average_rewards = {epsilon: np.zeros(n_plays) for epsilon in epsilons}
optimal_action_perc = {epsilon: np.zeros(n_plays) for epsilon in epsilons}

# Simulate each epsilon-greedy strategy
for epsilon in epsilons:
    rewards = np.zeros((n_tasks, n_plays))
    optimal_actions = np.zeros((n_tasks, n_plays))

    for task in range(n_tasks):
        q_estimates = np.zeros(n_bandits)  # Initial action-value estimates
        action_counts = np.zeros(n_bandits)  # Count of actions taken
        optimal_action = np.argmax(true_action_values[task])  # Optimal action for this task

        for play in range(n_plays):
            # Select action using epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(n_bandits)  # Random action (exploration)
            else:
                action = np.argmax(q_estimates)  # Greedy action (exploitation)

            # Track if the optimal action was chosen
            optimal_actions[task, play] = (action == optimal_action)

            # Get reward from the selected action
            reward = np.random.normal(true_action_values[task, action], 1)
            rewards[task, play] = reward

            # Update action-value estimate using incremental formula
            action_counts[action] += 1
            q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

    # Compute average reward and optimal action percentage for this epsilon
    average_rewards[epsilon] = rewards.mean(axis=0)
    optimal_action_perc[epsilon] = optimal_actions.mean(axis=0) * 100

# Plot Average Rewards
plt.figure(figsize=(12, 6))
for epsilon, rewards in average_rewards.items():
    plt.plot(rewards, label=f"$\\epsilon = {epsilon}$")
plt.xlabel("Plays")
plt.ylabel("Average Reward")
plt.title("Average Reward for $\\epsilon$-greedy Strategies")
plt.legend()
plt.grid()
plt.show()

# Plot Optimal Action Percentage
plt.figure(figsize=(12, 6))
for epsilon, perc in optimal_action_perc.items():
    plt.plot(perc, label=f"$\\epsilon = {epsilon}$")
plt.xlabel("Plays")
plt.ylabel("% Optimal Action")
plt.title("Percentage of Optimal Action for $\\epsilon$-greedy Strategies")
plt.legend()
plt.grid()
plt.show()
