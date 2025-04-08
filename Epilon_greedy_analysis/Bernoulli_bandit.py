import numpy as np
import matplotlib.pyplot as plt

# Set the parameters for the experiment
num_arms = 10
num_problems = 1500
num_steps = 1000
epsilons = [0, 0.01, 0.1]
c = 2  # UCB1 exploration parameter

# Function to simulate a single Bernoulli bandit problem
def simulate_bandit_problem(num_arms):
    return np.random.uniform(0, 1, num_arms)

# Epsilon-greedy algorithm
def epsilon_greedy(epsilon, bandit_probs, num_steps):
    num_arms = len(bandit_probs)
    q_values = np.zeros(num_arms)  # Estimated values
    arm_counts = np.zeros(num_arms)  # Number of times each arm is pulled
    total_reward = 0
    rewards = []

    for t in range(num_steps):
        if np.random.rand() < epsilon:
            arm = np.random.choice(num_arms)
        else:
            arm = np.argmax(q_values)
        
        reward = np.random.binomial(1, bandit_probs[arm])
        total_reward += reward
        rewards.append(total_reward / (t + 1))

        # Update the estimates
        arm_counts[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / arm_counts[arm]

    return rewards

# UCB1 algorithm
def ucb1(bandit_probs, num_steps, c):
    num_arms = len(bandit_probs)
    q_values = np.zeros(num_arms)  # Estimated values
    arm_counts = np.zeros(num_arms) + 1e-5  # Add a small value to avoid division by zero
    total_reward = 0
    rewards = []

    for t in range(num_steps):
        ucb_values = q_values + c * np.sqrt(np.log(t + 1) / arm_counts)
        arm = np.argmax(ucb_values)
        
        reward = np.random.binomial(1, bandit_probs[arm])
        total_reward += reward
        rewards.append(total_reward / (t + 1))

        # Update the estimates
        arm_counts[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / arm_counts[arm]

    return rewards

# Run the experiment
average_rewards = {epsilon: np.zeros(num_steps) for epsilon in epsilons}
average_rewards['UCB1'] = np.zeros(num_steps)

for _ in range(num_problems):
    bandit_probs = simulate_bandit_problem(num_arms)
    
    for epsilon in epsilons:
        rewards = epsilon_greedy(epsilon, bandit_probs, num_steps)
        average_rewards[epsilon] += rewards

    rewards = ucb1(bandit_probs, num_steps, c)
    average_rewards['UCB1'] += rewards

# Average the rewards
for key in average_rewards:
    average_rewards[key] /= num_problems

# Plot the results
plt.figure(figsize=(12, 8))
for epsilon in epsilons:
    plt.plot(average_rewards[epsilon], label=f"Epsilon-Greedy (epsilon={epsilon})")
plt.plot(average_rewards['UCB1'], label="UCB1", linestyle='--')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Comparison of Epsilon-Greedy and UCB1 in 10-Armed Bernoulli Bandit")
plt.legend()
plt.show()
