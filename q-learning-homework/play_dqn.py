import gymnasium as gym
import torch
from dqn_agent import DQNAgent

def play_trained_model():
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.policy_net.load_state_dict(torch.load("lunar_lander_dqn.pth"))
    agent.policy_net.eval()
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    play_trained_model()
