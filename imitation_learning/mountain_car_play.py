import gym
import pygame
from teleop import play

# Define key mappings for actions
mapping = {
    (pygame.K_LEFT,): 0,
    (pygame.K_RIGHT,): 2,
    (): 1  # Coast (no keys pressed)
}

# Initialize the environment with render_mode='rgb_array'
env = gym.make("MountainCar-v0", render_mode='rgb_array')

# Start the interactive play session
demos = play(env, keys_to_action=mapping)
