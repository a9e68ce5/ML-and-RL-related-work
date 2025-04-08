import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.optim import Adam
from core import MLPActorCritic  # 使用你提供的 actor-critic 架構
import matplotlib.pyplot as plt

def train(env_name='CartPole-v1', hidden_sizes=[32], 
          pi_lr=1e-3, vf_lr=1e-3, epochs=10, batch_size=2000, render=False):
    """
    使用 generalized advantage estimation (GAE) 搭配學習後的 value function 作為 baseline，
    訓練策略網路。此函式回傳每個 epoch 的平均回報以及訓練好的 actor-critic 模型 (ac)。
    """
    env = gym.make(env_name)
    # 利用 gym 提供的空間資訊建立 actor-critic (注意不要傳入整數維度)
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hidden_sizes, activation=nn.Tanh)
    
    # 分別建立策略與 value function 的 optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    
    gamma = 0.99      # 折扣因子
    lam = 0.97        # GAE lambda
    vf_iters = 80     # 每個 epoch 更新 value function 的次數
    
    avg_returns = []
    
    def train_one_epoch():
        # 用來存放軌跡數據的容器
        batch_obs = []
        batch_acts = []
        batch_rewards = []
        batch_dones = []
        batch_values = []  # 存放每個 state 的 V(s)
        episode_rets = []  # 記錄每一回合的總回報
        ep_rews = []       # 當前回合的回報暫存
        
        steps = 0
        obs, info = env.reset()
        # 收集資料直到步數達到 batch_size
        while steps < batch_size:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            # 使用 ac.step()，它會回傳 (action, value, logp) (皆為 numpy array)
            action, value, logp = ac.step(obs_tensor)
            batch_obs.append(obs.copy())
            batch_acts.append(action)
            batch_values.append(value)  # 記錄目前狀態的 value
            
            # 執行動作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            batch_rewards.append(reward)
            batch_dones.append(float(done))
            ep_rews.append(reward)
            steps += 1
            
            if done:
                # 回合結束，記錄該回合回報
                episode_rets.append(sum(ep_rews))
                ep_rews = []
                # 當回合結束，重設環境；此時不再立即新增 value (在下方統一處理 bootstrap)
                obs, info = env.reset()
        # 收集完 batch 後，對最後一個狀態做 bootstrap
        if batch_dones[-1] == 1.0:
            final_value = 0.0
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                final_value = ac.v(obs_tensor).item()
        # 轉換 batch_values 為 numpy array 並新增最後 bootstrap value (確保長度為 T+1)
        batch_values = np.array(batch_values, dtype=np.float32)
        batch_values = np.append(batch_values, final_value)
        
        return (np.array(batch_obs), 
                np.array(batch_acts), 
                np.array(batch_rewards, dtype=np.float32), 
                np.array(batch_dones, dtype=np.float32), 
                batch_values, 
                episode_rets)
    
    for epoch in range(epochs):
        # 收集一個 epoch 的軌跡數據
        batch_obs, batch_acts, rewards_np, dones_np, values_np, episode_rets = train_one_epoch()
        T = len(rewards_np)  # 該批次的總步數
        
        # 計算 GAE advantage
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones_np[t]
            delta = rewards_np[t] + gamma * values_np[t+1] * non_terminal - values_np[t]
            advantages[t] = delta + gamma * lam * non_terminal * last_adv
            last_adv = advantages[t]
        
        # 計算 reward-to-go (returns)
        returns = advantages + values_np[:-1]  # 長度 T
        
        # 將 advantage 正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 轉換資料為 torch tensor
        batch_obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        batch_acts_tensor = torch.as_tensor(batch_acts)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        
        # --------------------- 策略更新 ---------------------
        pi_optimizer.zero_grad()
        # 重新計算動作的 log probability
        _, logp = ac.pi(batch_obs_tensor, batch_acts_tensor)
        pi_loss = -(logp * advantages_tensor).mean()
        pi_loss.backward()
        pi_optimizer.step()
        
        # ------------------- Value function 更新 -------------------
        for _ in range(vf_iters):
            vf_optimizer.zero_grad()
            v_pred = ac.v(batch_obs_tensor).squeeze()
            vf_loss = nn.MSELoss()(v_pred, returns_tensor)
            vf_loss.backward()
            vf_optimizer.step()
        
        avg_ret = np.mean(episode_rets) if len(episode_rets) > 0 else 0.0
        avg_returns.append(avg_ret)
        print(f"Epoch: {epoch:3d} | Policy Loss: {pi_loss.item():.3f} | Value Loss: {vf_loss.item():.3f} | Avg Return: {avg_ret:.3f}")
        
        # 若 render 設為 True，則觀看一個回合的執行過程
        if render:
            obs, info = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    action = ac.pi._distribution(obs_tensor).sample().item()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
    env.close()
    return avg_returns, ac

def test_trained_model(ac, env_name="CartPole-v1", num_episodes=5):
    """
    利用訓練完成的 actor-critic 模型 (ac)
    在 gym 環境中執行 num_episodes 個回合，顯示畫面並印出每回合回報。
    """
    # 使用 render_mode="human" 讓 gym 顯示畫面
    env = gym.make(env_name, render_mode="human")
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = ac.pi._distribution(obs_tensor).sample().item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        print(f"Episode {i+1}: Return = {ep_ret:.3f}")
    env.close()

if __name__ == '__main__':
    # ----------------------- CartPole 測試 (離散動作) -----------------------
    print("=== Training on CartPole-v1 ===")
    avg_returns_cartpole, ac_cartpole = train(env_name='CartPole-v1', hidden_sizes=[32], 
                                              pi_lr=1e-3, vf_lr=1e-3, epochs=10, batch_size=2000, render=False)
    print("=== Testing on CartPole-v1 ===")
    test_trained_model(ac_cartpole, env_name="CartPole-v1", num_episodes=5)

    # ----------------------- Pendulum 測試 (連續動作) -----------------------
    # Pendulum 環境需要連續動作，因此使用 MLPGaussianActor
    print("=== Training on Pendulum-v1 ===")
    # 由於 Pendulum 問題較簡單，這裡可以適度調整參數，例如使用較大的 hidden layer 與不同的學習率
    avg_returns_pendulum, ac_pendulum = train(env_name='Pendulum-v1', hidden_sizes=[64], 
                                              pi_lr=3e-4, vf_lr=1e-3, epochs=30, batch_size=5000, render=False)
    print("=== Testing on Pendulum-v1 ===")
    test_trained_model(ac_pendulum, env_name="Pendulum-v1", num_episodes=5)
