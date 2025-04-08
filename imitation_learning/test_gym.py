import gym

# Create the environment
env = gym.make('MountainCar-v0', render_mode='human')

# Reset the environment
env.reset()

done = False
while not done:
    # Take a random action
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

# Close the environment
env.close()
