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
    # 利用 gym 空間資訊建立 actor-critic 模型（注意不要傳入整數維度）
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hidden_sizes, activation=nn.Tanh)
    
    # 建立分別用於策略與 value function 更新的 optimizer
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    
    gamma = 0.99      # 折扣因子
    lam = 0.97        # GAE lambda
    vf_iters = 80     # 每個 epoch 更新 value function 次數
    
    avg_returns = []
    
    def train_one_epoch():
        # 用來儲存軌跡資料的容器
        batch_obs = []
        batch_acts = []
        batch_rewards = []
        batch_dones = []
        batch_values = []  # 儲存每個 state 的 V(s)
        episode_rets = []  # 儲存每回合總回報
        ep_rews = []       # 當前回合回報暫存
        
        steps = 0
        obs, info = env.reset()
        while steps < batch_size:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            # 使用 ac.step()，會回傳 (action, value, logp)
            action, value, logp = ac.step(obs_tensor)
            batch_obs.append(obs.copy())
            batch_acts.append(action)
            batch_values.append(value)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            batch_rewards.append(reward)
            batch_dones.append(float(done))
            ep_rews.append(reward)
            steps += 1
            
            if done:
                episode_rets.append(sum(ep_rews))
                ep_rews = []
                obs, info = env.reset()
        # 對最後狀態做 bootstrap
        if batch_dones[-1] == 1.0:
            final_value = 0.0
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                final_value = ac.v(obs_tensor).item()
        batch_values = np.array(batch_values, dtype=np.float32)
        # 確保 V(s) 序列長度為 T+1
        batch_values = np.append(batch_values, final_value)
        
        return (np.array(batch_obs), 
                np.array(batch_acts), 
                np.array(batch_rewards, dtype=np.float32), 
                np.array(batch_dones, dtype=np.float32), 
                batch_values, 
                episode_rets)
    
    for epoch in range(epochs):
        # 收集一個 epoch 的資料
        batch_obs, batch_acts, rewards_np, dones_np, values_np, episode_rets = train_one_epoch()
        T = len(rewards_np)
        
        # 計算 GAE advantage
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones_np[t]
            delta = rewards_np[t] + gamma * values_np[t+1] * non_terminal - values_np[t]
            advantages[t] = delta + gamma * lam * non_terminal * last_adv
            last_adv = advantages[t]
        # reward-to-go (returns) = advantage + V(s)
        returns = advantages + values_np[:-1]
        
        # 將 advantage 正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        batch_acts_tensor = torch.as_tensor(batch_acts)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        
        # --------------------- 策略更新 ---------------------
        pi_optimizer.zero_grad()
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
        
        if render:
            obs, info = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action = ac.pi._distribution(obs_tensor).sample().item()
                    else:
                        action = ac.pi._distribution(obs_tensor).sample().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
    env.close()
    return avg_returns, ac

def test_trained_model(ac, env_name="CartPole-v1", num_episodes=5):
    """
    利用訓練完成的 actor-critic 模型 (ac)
    在指定環境中執行 num_episodes 個回合，顯示畫面並印出每回合回報。
    """
    env = gym.make(env_name, render_mode="human")
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action_tensor = ac.pi._distribution(obs_tensor).sample()
            if is_discrete:
                action = action_tensor.item()
            else:
                action = action_tensor.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        print(f"Episode {i+1}: Return = {ep_ret:.3f}")
    env.close()

if __name__ == '__main__':
    # ----------------------- CartPole 測試 (離散動作) -----------------------
    print("=== Training on CartPole-v1 ===")
    # 設定參數
    cartpole_pi_lr = 1e-3
    cartpole_vf_lr = 1e-3
    cartpole_epochs = 300
    avg_returns_cartpole, ac_cartpole = train(env_name='CartPole-v1', hidden_sizes=[32], 
                                              pi_lr=cartpole_pi_lr, vf_lr=cartpole_vf_lr, 
                                              epochs=cartpole_epochs, batch_size=2000, render=False)
    print("=== Testing on CartPole-v1 ===")
    test_trained_model(ac_cartpole, env_name="CartPole-v1", num_episodes=5)
    
    # 畫出 CartPole 訓練曲線，並在圖上加上 learning rate 與 epoch 數
    plt.figure()
    plt.plot(avg_returns_cartpole, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title(f"Training Performance on CartPole-v1\nPolicy LR: {cartpole_pi_lr}, Value LR: {cartpole_vf_lr}, Epochs: {cartpole_epochs}")
    plt.grid(True)
    plt.show()
    
    # ----------------------- HalfCheetah 測試 (連續動作) -----------------------
    print("=== Training on HalfCheetah-v5 ===")
    halfcheetah_pi_lr = 3e-3
    halfcheetah_vf_lr = 1e-3
    halfcheetah_epochs = 300
    avg_returns_halfcheetah, ac_halfcheetah = train(env_name='HalfCheetah-v5', hidden_sizes=[64], 
                                                    pi_lr=halfcheetah_pi_lr, vf_lr=halfcheetah_vf_lr, 
                                                    epochs=halfcheetah_epochs, batch_size=5000, render=False)
    print("=== Testing on HalfCheetah-v5 ===")
    test_trained_model(ac_halfcheetah, env_name="HalfCheetah-v5", num_episodes=5)
    
    # 畫出 HalfCheetah 訓練曲線，並在圖上加上 learning rate 與 epoch 數
    plt.figure()
    plt.plot(avg_returns_halfcheetah, marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title(f"Training Performance on HalfCheetah-v5\nPolicy LR: {halfcheetah_pi_lr}, Value LR: {halfcheetah_vf_lr}, Epochs: {halfcheetah_epochs}")
    plt.grid(True)
    plt.show()

