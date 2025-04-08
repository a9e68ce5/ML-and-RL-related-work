import gymnasium as gym
import numpy as np
import random
import collections
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # 新增引入
import matplotlib.pyplot as plt  # 新增畫圖用的模組

# 設定 device (建議有 GPU 會比較好)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 離散化動作 Wrapper
# -------------------------------
class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    將 CarRacing-v2 的連續動作空間轉換為離散動作集合：
      0: 無動作
      1: 左轉 + 加速
      2: 右轉 + 加速
      3: 直行 + 加速
      4: 煞車
    """
    def __init__(self, env):
        super().__init__(env)
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
# Replay Memory
# -------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones
    
    def __len__(self):
        return len(self.memory)

# -------------------------------
# DQN 網路結構 (使用卷積層與全連接層)
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
# 預處理函數
# -------------------------------
def preprocess_obs(obs):
    """
    將觀測值轉為 float32、將像素值歸一化到 [0, 1]，
    並調整為 (C, H, W) 的順序。
    """
    obs = np.array(obs, dtype=np.float32) / 255.0
    # 如果觀測值 shape 為 (H, W, C)
    if obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))
    else:
        obs = np.expand_dims(obs, axis=0)
    return obs
# -------------------------------
# 超參數設定
# -------------------------------
num_episodes = 100           # 訓練回合數
max_steps = 1000             # 每回合最大步數
batch_size = 64
gamma = 0.99                 # 折扣因子
learning_rate = 1e-4
replay_memory_capacity = 100000
target_update_freq = 1000    # 每多少步更新一次 target network
start_training_after = 1000  # 收集多少步後開始訓練
epsilon_start = 1.0
epsilon_end = 0.02
epsilon_decay = 30000        # epsilon 線性衰減步數


# -------------------------------
# 建立環境與包裝器
# -------------------------------
env = gym.make("CarRacing-v3", render_mode=None)
env = DiscretizeActionWrapper(env)
env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
env = gym.wrappers.ResizeObservation(env, (84, 84))

num_actions = env.action_space.n
input_channels = 1

# -------------------------------
# 初始化網路、優化器與 Replay Memory
# -------------------------------
policy_net = DQN(input_channels, num_actions).to(device)
target_net = DQN(input_channels, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(replay_memory_capacity)

# -------------------------------
# epsilon-greedy 動作選擇
# -------------------------------
epsilon = epsilon_start
step_count = 0

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # shape: (1, C, H, W)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()

# 用來儲存每個回合的總 reward，供學習曲線使用
episode_rewards = []

# -------------------------------
# 主訓練迴圈
# -------------------------------
for episode in range(num_episodes):
    obs, info = env.reset()
    state = preprocess_obs(obs)
    total_reward = 0
    
    for t in range(max_steps):
        step_count += 1
        action = select_action(state, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = preprocess_obs(next_obs)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # 線性衰減 epsilon
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
        
        # 當記憶庫有足夠樣本且超過起始訓練步數後，進行訓練
        if len(memory) > batch_size and step_count > start_training_after:
            states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
            states_b = torch.from_numpy(states_b).to(device)
            actions_b = torch.tensor(actions_b, device=device).unsqueeze(1)
            rewards_b = torch.tensor(rewards_b, device=device).unsqueeze(1)
            next_states_b = torch.from_numpy(next_states_b).to(device)
            dones_b = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)
            
            # 計算當前 Q 值及 target Q 值
            q_values = policy_net(states_b).gather(1, actions_b)
            with torch.no_grad():
                next_q_values = target_net(next_states_b).max(1)[0].unsqueeze(1)
            expected_q_values = rewards_b + gamma * next_q_values * (1 - dones_b)
            
            # 使用 smooth L1 loss 來計算損失
            loss = F.smooth_l1_loss(q_values.squeeze(-1), expected_q_values.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #loss = nn.MSELoss()(q_values, expected_q_values)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
        
        # 定期更新 target network
        if step_count % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# 儲存模型權重
torch.save(policy_net.state_dict(), "dqn_car_racing.pth")
print("Model has been saved")

# -------------------------------
# 畫出學習曲線 (Learning Curve)
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes+1), episode_rewards, label="Total Reward per Episode", marker='o', markersize=3, linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve for DQN on Car Racing")
plt.legend()
plt.grid(True)
plt.savefig("learning_curve.png")
plt.show()
