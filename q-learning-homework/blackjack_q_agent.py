import gymnasium as gym
import numpy as np
import pickle

# Load trained Q-table
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

# Create environment
env = gym.make("Blackjack-v1", render_mode="human")

# Function to play using trained Q-table
def play_blackjack():
    state, _ = env.reset()
    done = False
    
    while not done:
        action = np.argmax(Q.get(state, np.zeros(env.action_space.n)))  # Select best action
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
    
    # Print result
    if reward == 1:
        print("\nYou won!")
    elif reward == -1:
        print("\nYou lost!")
    else:
        print("\nIt's a draw!")

    env.close()

# Play a game with the trained agent
play_blackjack()
