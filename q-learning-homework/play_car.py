import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------
# DQN 網路結構 (與 train.py 相同)
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # 計算卷積層輸出尺寸 (84x84 輸入)
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = convw  # 輸入為正方形
        linear_input_size = convw * convh * 64
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------
# 預處理函數 (與 train.py 相同)
# -------------------------------
def preprocess_obs(obs):
    obs = np.array(obs, dtype=np.float32) / 255.0
    if obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))
    else:
        obs = np.expand_dims(obs, axis=0)
    return obs

# -------------------------------
# 離散化連續動作空間 (與 train.py 相同)
# -------------------------------
class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.discrete_actions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),    # 無動作
            np.array([-1.0, 1.0, 0.0], dtype=np.float32),     # 左轉 + 加速
            np.array([1.0, 1.0, 0.0], dtype=np.float32),      # 右轉 + 加速
            np.array([0.0, 1.0, 0.0], dtype=np.float32),      # 直行 + 加速
            np.array([0.0, 0.0, 0.8], dtype=np.float32)       # 煞車
        ]
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
    
    def action(self, action):
        return self.discrete_actions[action]

# -------------------------------
# 設定裝置與環境
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立環境 (不啟用 render_mode，方便批次測試)
env = gym.make("CarRacing-v3", render_mode="human")
env = DiscretizeActionWrapper(env)
env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
env = gym.wrappers.ResizeObservation(env, (84, 84))

num_actions = env.action_space.n
input_channels = 1

# -------------------------------
# 載入訓練好的模型
# -------------------------------
policy_net = DQN(input_channels, num_actions).to(device)
policy_net.load_state_dict(torch.load("dqn_car_racing.pth", map_location=device))
policy_net.eval()

# -------------------------------
# 定義評估函數
# -------------------------------
def evaluate_agent(env, policy_net, num_episodes=10):
    rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = preprocess_obs(obs)
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = preprocess_obs(next_obs)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Agent Episode {episode+1}: Total Reward = {total_reward:.2f}")
    return rewards

def evaluate_random(env, num_episodes=10):
    rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"Random Episode {episode+1}: Total Reward = {total_reward:.2f}")
    return rewards

# -------------------------------
# 執行評估並收集結果
# -------------------------------
num_eval_episodes = 20
agent_rewards = evaluate_agent(env, policy_net, num_episodes=num_eval_episodes)
random_rewards = evaluate_random(env, num_episodes=num_eval_episodes)

env.close()

# -------------------------------
# 畫圖比較結果
# -------------------------------
episodes = list(range(1, num_eval_episodes + 1))
plt.figure(figsize=(10, 5))
plt.plot(episodes, agent_rewards, label="Agent", marker='o')
plt.plot(episodes, random_rewards, label="Random", marker='o')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Agent vs Random Policy Performance on Car Racing")
plt.legend()
plt.grid(True)
plt.savefig("agent_vs_random.png")
plt.show()
