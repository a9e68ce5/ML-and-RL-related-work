import gymnasium as gym
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# --------------------- 原本的 Q-Learning + UCB 實作 ---------------------

# 建立 Blackjack 環境 (Blackjack-v1 為 Gymnasium 提供的 Blackjack 版本)
env = gym.make("Blackjack-v1")

def ucb_action_selection(Q, state, N, episode):
    """
    使用上界置信區間 (UCB, Upper Confidence Bound) 策略來選擇動作。
    根據 Q 值與探索獎勵 (exploration bonus) 的結合進行選擇，鼓勵尚未嘗試過的動作。

    :param Q: Q 表 (預期回報)，型態為 defaultdict，每個 state 對應一個動作值陣列
    :param state: 當前狀態
    :param N: 每個狀態的訪問次數記錄 (同樣為 defaultdict)，用來計算探索獎勵
    :param episode: 當前的回合數，用來計算對數項 (log(episode+1))
    :return: 返回具有最大 (Q + exploration bonus) 的動作索引
    """
    # 為了避免除以 0，這裡加上一個很小的值 1e-6
    exploration = math.sqrt(math.log(episode + 1) / (N[state] + 1e-6))
    # np.argmax 選出具有最高 (Q[state] + exploration) 的動作
    return np.argmax(Q[state] + exploration)

# 初始化 Q 表 (使用 defaultdict，自動為每個新狀態建立一個全 0 陣列)
default_Q = lambda: np.zeros(env.action_space.n)
Q = defaultdict(default_Q)

# 初始化記錄各狀態訪問次數的字典 (N[state] 為該狀態被造訪的次數)
N = defaultdict(int)

# --------------------- 超參數設定 ---------------------
alpha = 0.0000001  # 學習率 (越大更新越快，但可能不穩定；此處設定偏小)
gamma = 0.995      # 折扣因子，鼓勵長期獎勵
epsilon = 1.0      # 初始 epsilon (探索率) 值，初期完全隨機探索
epsilon_decay = 0.99997  # 探索率衰減速率，用於動態降低隨機探索比例
epsilon_min = 0.01       # 探索率的下限，保證一定程度的隨機性
num_episodes = 400000    # 總訓練回合數 (集數)，數量較多有助於學習
moving_avg_window = 2000  # 用於平滑曲線的移動平均窗口大小

# --------------------- 用來追蹤訓練進度的變數 ---------------------
time_steps = []         # 記錄每個回合的編號
cumulative_rewards = [] # 累計平均獎勵 (隨著回合增加會漸進變化)
average_rewards = []    # 儲存每次評估的平均獎勵
eval_window = 5000      # 每隔多少回合進行一次策略評估

def evaluate_policy(Q, num_episodes=6000):
    """
    評估目前的策略 (Q 表) 表現，透過在環境中運行多個回合統計勝、負、平局情形

    :param Q: 用來評估的 Q 表 (字典)
    :param num_episodes: 評估的回合數
    :return: 一個四元組 (勝利數, 失敗數, 平局數, 勝率)
    """
    win, loss, draw = 0, 0, 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # 若狀態不存在於 Q 表中，則回傳全 0 陣列
            action = np.argmax(Q.get(state, np.zeros(env.action_space.n)))
            state, reward, done, _, _ = env.step(action)
            # 統計根據 reward 區分勝利、失敗與平局
            if reward == 1:
                win += 1
            elif reward == -1:
                loss += 1
            else:
                draw += 1
    return (win, loss, draw, win / num_episodes)

def moving_average(data, window=1000):
    """
    計算移動平均，用來平滑獎勵變化的曲線

    :param data: 數據列表
    :param window: 移動平均的窗口大小
    :return: 計算後的移動平均數據 (numpy 陣列)
    """
    return np.convolve(data, np.ones(window)/window, mode='valid')

# 用來記錄隨機策略和 Q-Learning 策略的評估結果 (用於後續比較)
random_policy_rewards = []
q_learning_rewards = []

# --------------------- Q-Learning 訓練迴圈 ---------------------
for episode in range(num_episodes):
    # 每回合開始時，重置環境
    state, _ = env.reset()
    done = False
    episode_reward = 0  # 初始化本回合的累計獎勵
    
    # 當前回合尚未結束，持續進行以下步驟
    while not done:
        # epsilon-greedy 探索：以 epsilon 機率採用隨機動作
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # 否則使用 UCB 策略從 Q 表中選擇動作
            action = ucb_action_selection(Q, state, N, episode)
        
        # 執行所選動作，獲得下個狀態、獎勵以及是否結束的訊息
        next_state, reward, done, _, _ = env.step(action)
        
        # --------------------- 中間獎勵塑形 ---------------------
        # 當玩家手牌總點數大於等於 18 時，額外增加 0.1 的獎勵，以鼓勵玩家累積強牌
        if state[0] >= 18:
            reward += 0.1  
        # 如果玩家選擇要牌 (action == 1) 且手牌總點數超過 19，則扣除 0.1 的獎勵 (避免過度冒險)
        if action == 1 and state[0] > 19:
            reward -= 0.1
            
        # --------------------- Q-Learning 更新規則 ---------------------
        # 選擇下一狀態中最佳的動作（利用 Q 表中的最大值）
        best_next_action = np.argmax(Q.get(next_state, np.zeros(env.action_space.n)))
        # 使用 Q-Learning 更新公式：Q(s,a) <- Q(s,a) + alpha * [reward + gamma * Q(s', a*) - Q(s,a)]
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
        
        # 將狀態更新為下一狀態，並累計獎勵
        state = next_state
        episode_reward += reward
        # 更新當前狀態的訪問次數，這個計數會用在 UCB 的探索計算中
        N[state] += 1  
    
    # --------------------- 每隔一定回合數進行策略評估 ---------------------
    if episode % eval_window == 0:
        win, loss, draw, avg_reward = evaluate_policy(Q)
        q_learning_rewards.append(avg_reward)
        print(f"Episode {episode}: Wins={win}, Losses={loss}, Draws={draw}, Win Rate={avg_reward:.2f}")
        # 評估隨機策略的勝率 (使用 default_Q 來產生全零 Q 表)
        random_policy_rewards.append(evaluate_policy(defaultdict(default_Q))[3])
        average_rewards.append(avg_reward)
    
    # 紀錄回合數與累計平均獎勵 (這裡 cumulative_rewards 為所有評估的平均獎勵取均值)
    time_steps.append(episode)
    cumulative_rewards.append(sum(average_rewards) / len(average_rewards))
    
    # --------------------- 動態衰減 epsilon ---------------------
    # 這裡使用線性衰減的方式，隨著回合數增加，epsilon 逐漸下降，但不低於 epsilon_min
    epsilon = max(epsilon_min, epsilon * (1 - (episode / num_episodes)))

# --------------------- 訓練結束後保存 Q 表 ---------------------
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

# --------------------- 繪製學習曲線 ---------------------
# 使用移動平均平滑累計獎勵曲線
plt.figure(figsize=(10, 5))
plt.plot(time_steps[moving_avg_window-1:], moving_average(cumulative_rewards, moving_avg_window),
         label='Q-Learning Agent', color='blue')
plt.xlabel("Episodes")
plt.ylabel("Smoothed Cumulative Reward")
plt.title("Learning Curve of Q-Learning Agent in Blackjack")
plt.legend()
plt.show()

# 繪製 Q-Learning 策略與隨機策略的比較圖
plt.figure(figsize=(10, 5))
plt.plot(range(0, num_episodes, eval_window), q_learning_rewards, label='Q-Learning Policy', marker='o')
plt.plot(range(0, num_episodes, eval_window), random_policy_rewards, label='Random Policy', marker='s', linestyle='dashed')
plt.xlabel("Episodes")
plt.ylabel("Win Rate over 6000 episodes")
plt.title("Performance Comparison: Q-Learning vs Random Policy")
plt.legend()
plt.show()

print("Training complete! Q-table saved.")


# --------------------- 額外實驗：比較 Boltzmann 探索與不同 epsilon 的 epsilon-greedy ---------------------

# 為了方便比較，以下將 Q-Learning 的訓練迴圈包成兩個函數，
# 分別實作 epsilon-greedy 探索（不含 UCB）以及 Boltzmann（softmax）探索策略。
# 注意：num_episodes_experiment 可以根據需求調整 (此處設定為 100000 集數)。

num_episodes_experiment = 100000
eval_window_experiment = 10000  # 每隔這麼多集數進行一次策略評估

def train_agent_eps_greedy(initial_epsilon, epsilon_decay, epsilon_min, num_episodes, eval_window):
    """
    用純 epsilon-greedy 探索策略進行 Q-Learning 訓練，不採用 UCB 探索。
    
    :param initial_epsilon: 初始探索率 epsilon
    :param epsilon_decay: epsilon 衰減係數
    :param epsilon_min: epsilon 的最小值
    :param num_episodes: 總訓練回合數
    :param eval_window: 每隔多少回合進行一次策略評估
    :return: 訓練完後的 Q 表、評估回合數列表與每次評估的平均獎勵歷史
    """
    # 初始化 Q 表，每個新狀態的預設動作值皆為 0
    Q_eps = defaultdict(lambda: np.zeros(env.action_space.n))
    eval_history = []  # 用來記錄每次評估的平均勝率
    time_steps = []    # 用來記錄評估時的回合數
    epsilon_local = initial_epsilon  # 使用局部變數追蹤 epsilon 的動態變化
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # epsilon-greedy 策略：若隨機數小於 epsilon，則隨機選擇動作
            if random.uniform(0, 1) < epsilon_local:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_eps[state])
            
            next_state, reward, done, _, _ = env.step(action)
            
            # 與原本相同的 reward shaping：根據手牌總點數與行動調整獎勵
            if state[0] >= 18:
                reward += 0.1
            if action == 1 and state[0] > 19:
                reward -= 0.1
                
            best_next_action = np.argmax(Q_eps[next_state])
            # Q-Learning 更新
            Q_eps[state][action] += alpha * (reward + gamma * Q_eps[next_state][best_next_action] - Q_eps[state][action])
            state = next_state
        
        # 每隔 eval_window 回合評估一次目前策略
        if episode % eval_window == 0:
            _, _, _, avg_reward = evaluate_policy(Q_eps)
            eval_history.append(avg_reward)
            time_steps.append(episode)
        
        # 衰減 epsilon：隨著回合增加，逐步降低隨機探索比例
        epsilon_local = max(epsilon_min, epsilon_local * (1 - (episode / num_episodes)))
    
    return Q_eps, time_steps, eval_history


def train_agent_boltzmann(initial_temperature, temperature_decay, num_episodes, eval_window):
    """
    使用 Boltzmann (softmax) 探索策略進行 Q-Learning 訓練。
    
    :param initial_temperature: 初始溫度參數，控制 softmax 的隨機性
    :param temperature_decay: 溫度衰減係數，每回合將溫度逐漸降低
    :param num_episodes: 總訓練回合數
    :param eval_window: 每隔多少回合進行一次策略評估
    :return: 訓練後的 Q 表、評估回合數列表與評估歷史
    """
    # 初始化 Q 表，預設值皆為 0
    Q_boltz = defaultdict(lambda: np.zeros(env.action_space.n))
    eval_history = []  # 用來記錄每次策略評估的平均勝率
    time_steps = []    # 評估時所對應的回合數
    temperature = initial_temperature  # 初始化溫度參數
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # 取出當前狀態的 Q 值陣列
            q_values = Q_boltz[state]
            # 為了數值穩定，先從 q_values 中減去最大值
            max_q = np.max(q_values)
            # 計算 softmax 的分子部分：exp((q - max_q) / temperature)
            exp_q = np.exp((q_values - max_q) / temperature)
            # 計算概率分布：將 exp 值正規化
            probs = exp_q / np.sum(exp_q)
            # 根據 softmax 概率分布隨機選擇動作
            action = np.random.choice(range(env.action_space.n), p=probs)
            
            next_state, reward, done, _, _ = env.step(action)
            
            # 同樣進行 reward shaping 處理
            if state[0] >= 18:
                reward += 0.1
            if action == 1 and state[0] > 19:
                reward -= 0.1
                
            best_next_action = np.argmax(Q_boltz[next_state])
            # Q-Learning 更新公式
            Q_boltz[state][action] += alpha * (reward + gamma * Q_boltz[next_state][best_next_action] - Q_boltz[state][action])
            state = next_state
        
        # 每隔 eval_window 回合進行一次策略評估
        if episode % eval_window == 0:
            _, _, _, avg_reward = evaluate_policy(Q_boltz)
            eval_history.append(avg_reward)
            time_steps.append(episode)
        
        # 衰減 temperature，降低隨機性，但不讓其低於 0.01
        temperature = max(0.01, temperature * temperature_decay)
    
    return Q_boltz, time_steps, eval_history


# --------------------- 執行實驗並繪圖 ---------------------

# 1. 測試 epsilon-greedy 策略，嘗試不同的初始 epsilon 值
epsilon_list = [1.0, 0.5, 0.1]
eval_results_eps = {}

for init_eps in epsilon_list:
    print(f"\nTraining with epsilon-greedy (initial epsilon = {init_eps}) ...")
    Q_eps, ts_eps, eval_hist_eps = train_agent_eps_greedy(initial_epsilon=init_eps,
                                                           epsilon_decay=0.99997,
                                                           epsilon_min=0.01,
                                                           num_episodes=num_episodes_experiment,
                                                           eval_window=eval_window_experiment)
    eval_results_eps[init_eps] = (ts_eps, eval_hist_eps)

# 繪製不同初始 epsilon 值下的 epsilon-greedy 策略表現曲線
plt.figure(figsize=(10, 5))
for init_eps in epsilon_list:
    ts, eval_hist = eval_results_eps[init_eps]
    plt.plot(ts, eval_hist, marker='o', label=f'epsilon-greedy, init eps={init_eps}')
plt.xlabel("Episodes")
plt.ylabel("Win Rate over 6000 episodes")
plt.title("Performance of epsilon-greedy Exploration")
plt.legend()
plt.show()


# 2. 測試 Boltzmann 探索策略，嘗試不同的初始 temperature 值
temperature_list = [1.0, 0.5, 0.1]
eval_results_boltz = {}

for init_temp in temperature_list:
    print(f"\nTraining with Boltzmann exploration (initial temperature = {init_temp}) ...")
    Q_boltz, ts_boltz, eval_hist_boltz = train_agent_boltzmann(initial_temperature=init_temp,
                                                                 temperature_decay=0.9999,
                                                                 num_episodes=num_episodes_experiment,
                                                                 eval_window=eval_window_experiment)
    eval_results_boltz[init_temp] = (ts_boltz, eval_hist_boltz)

# 繪製不同初始 temperature 值下的 Boltzmann 探索策略表現曲線
plt.figure(figsize=(10, 5))
for init_temp in temperature_list:
    ts, eval_hist = eval_results_boltz[init_temp]
    plt.plot(ts, eval_hist, marker='o', label=f'Boltzmann, init temp={init_temp}')
plt.xlabel("Episodes")
plt.ylabel("Win Rate over 6000 episodes")
plt.title("Performance of Boltzmann (Softmax) Exploration")
plt.legend()
plt.show()
