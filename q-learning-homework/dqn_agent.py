import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

# 定义深度 Q 网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 定义 DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.memory = ReplayMemory(50000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.steps_done = 0
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # LunarLander-v3 有4个离散动作
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64).to(self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.bool).to(self.device)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))
        
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# 绘制学习曲线
def plot_learning_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig("learning_curve.png")
    plt.show()

# 训练 DQN Agent（包含平稳着陆的奖励塑形）
def train_dqn(num_episodes=3000):
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(state_dim=8, action_dim=4)
    
    rewards = []  # 用于绘图的奖励记录

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 平稳着陆奖励塑形:
            # LunarLander-v3 的状态向量为:
            # [x, y, x_dot, y_dot, angle, angular_velocity, left_leg_contact, right_leg_contact]
            if terminated:
                vertical_speed = abs(next_state[3])
                angular_speed = abs(next_state[5])
                vertical_threshold = 2.0   # 平稳着陆所需的最大垂直速度
                angular_threshold = 0.1    # 平稳着陆所需的最大角速度

                if vertical_speed > vertical_threshold or angular_speed > angular_threshold:
                    # 着陆不平稳或坠毁：施加惩罚
                    reward -= 100  
                else:
                    # 着陆平稳：给予奖励
                    reward += 100  

            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            agent.optimize_model()
        
        agent.decay_epsilon()
        
        if episode % agent.target_update == 0:
            agent.update_target_net()
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    torch.save(agent.policy_net.state_dict(), "lunar_lander_dqn.pth")
    print("模型已保存！")

    # 绘制学习曲线
    plot_learning_curve(rewards)

# 使用训练好的模型进行游戏
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
    
    print(f"总奖励: {total_reward:.2f}")
    env.close()

# 对策略进行评估，返回各集奖励列表
def evaluate_policy_rewards(env, agent, episodes=10, random_policy=False):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action = agent.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards

# 可视化随机策略与 DQN 策略的比较
def visualize_policy_comparison(random_rewards, dqn_rewards):
    plt.figure(figsize=(10, 6))
    plt.boxplot([random_rewards, dqn_rewards], labels=["Random Policy", "DQN Policy"])
    plt.ylabel("Total Reward per Episode")
    plt.title("Comparison of Random Policy vs. DQN Policy")
    plt.grid()
    plt.savefig("policy_comparison.png")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="使用训练好的模型进行游戏")
    args = parser.parse_args()
    
    if args.play:
        play_trained_model()
    else:
        # 先评估随机策略
        env = gym.make("LunarLander-v3")
        random_rewards = evaluate_policy_rewards(env, None, episodes=10, random_policy=True)
        random_avg = np.mean(random_rewards)
        random_std = np.std(random_rewards)
        print(f"随机策略 -> 平均奖励: {random_avg:.2f}, 标准差: {random_std:.2f}")
        env.close()
        
        # 训练 DQN 策略（包含平稳着陆的奖励塑形）
        train_dqn(1000)
        
        # 评估训练后的 DQN 策略
        env = gym.make("LunarLander-v3")
        agent = DQNAgent(state_dim=8, action_dim=4)
        agent.policy_net.load_state_dict(torch.load("lunar_lander_dqn.pth"))
        agent.policy_net.eval()
        dqn_rewards = evaluate_policy_rewards(env, agent, episodes=10, random_policy=False)
        dqn_avg = np.mean(dqn_rewards)
        dqn_std = np.std(dqn_rewards)
        print(f"DQN策略 -> 平均奖励: {dqn_avg:.2f}, 标准差: {dqn_std:.2f}")
        env.close()
        
        # 可视化比较结果
        visualize_policy_comparison(random_rewards, dqn_rewards)
