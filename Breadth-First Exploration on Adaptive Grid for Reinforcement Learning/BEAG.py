# BEAG_Antv4.py - 完整版 (含 HER、Policy、Replay Buffer + Algo 6 + 可視化、成功率、checkpoint)
import gymnasium as gym
import numpy as np
import heapq
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os

# --- 訓練參數 ---
nepi = 500
nrand = 50
nref = 20
T = 500
rt = 100
tau_n = 3

# --- 環境與狀態維度 ---
env = gym.make('Ant-v5')
phi = lambda s: s[:2]  # 假設前兩維為 (x, y)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

delta0 = 1.0
grid_L = -5.0
grid_U = 5.0

gamma = 0.99
lr = 1e-3
buffer_size = 100000
batch_size = 128
h_max = 50

# --- 資料夾建立 ---
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# --- 成功率記錄 ---
success_rates = []

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, g, a, r, s_next, done):
        self.buffer.append((s, g, a, r, s_next, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, g, a, r, s_next, d = map(np.array, zip(*batch))
        return map(lambda x: torch.tensor(x, dtype=torch.float32), (s, g, a, r, s_next, d))
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer(buffer_size)

# --- Policy ---
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 2, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim + 2 + action_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
    def act(self, s, g):
        x = torch.cat([s, g], dim=-1)
        return self.actor(x)
    def Q(self, s, g, a):
        x = torch.cat([s, g, a], dim=-1)
        return self.critic(x)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=lr)

# --- Grid ---
def construct_grid_graph(delta):
    V = []
    E = []
    coords = np.arange(grid_L, grid_U + delta, delta)
    for x in coords:
        for y in coords:
            V.append((x, y))
    for u in V:
        for v in V:
            if np.linalg.norm(np.array(u) - np.array(v)) == delta:
                E.append((u, v))
    return set(V), set(E)

def refine_grid(V, E, delta):
    new_V = set(V)
    new_E = set(E)
    for v in list(V):
        neighbors = [w for (u, w) in E if u == v]
        for w in neighbors:
            mid = ((v[0] + w[0]) / 2, (v[1] + w[1]) / 2)
            if mid not in new_V:
                new_V.add(mid)
                new_E.add((v, mid))
                new_E.add((mid, w))
                new_E.discard((v, w))
    return new_V, new_E

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_path(start, goal, V, E):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for u, v in E:
            if u == current:
                new_cost = cost_so_far[u] + distance(u, v)
                if v not in cost_so_far or new_cost < cost_so_far[v]:
                    cost_so_far[v] = new_cost
                    priority = new_cost + distance(v, goal)
                    heapq.heappush(frontier, (priority, v))
                    came_from[v] = u
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return []
    path.append(start)
    path.reverse()
    return path

def find_closest_node(p, V):
    return min(V, key=lambda v: distance(p, v))

def find_path_with_fallback(start, goal, V, E):
    # 若 start 或 goal 不在 V 中，則找最近點
    start = find_closest_node(start, V) if tuple(np.round(start)) not in V else tuple(np.round(start))
    goal = find_closest_node(goal, V) if goal not in V else goal
    path = find_path(start, goal, V, E)
    if not path:
        path = [goal]
    return path

# --- Training Policy ---
def train_policy():
    if len(buffer) < batch_size:
        return
    s, g, a, r, s_next, d = buffer.sample(batch_size)
    target_Q = r + gamma * policy.Q(s_next, g, policy.act(s_next, g)).detach() * (1 - d)
    Q_val = policy.Q(s, g, a)
    loss_critic = ((Q_val - target_Q) ** 2).mean()
    a_pred = policy.act(s, g)
    loss_actor = -policy.Q(s, g, a_pred).mean()
    loss = loss_critic + loss_actor
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Main Loop ---
V, E = construct_grid_graph(delta0)
subgoal_failures = {v: 0 for v in V}

# refine 條件設定：當子目標失敗總數超過 grid 節點數的 25%，且目前節點數小於 max_V，則觸發 refine
max_V = 15000
def should_refine(subgoal_failures, V):
    return any(f > tau_n for f in subgoal_failures.values()) or (sum(subgoal_failures.values()) > 0.25 * len(V))

for ep in range(nepi):
    obs, info = env.reset()
    s = phi(obs)
    g = random.choice(list(V))

    # 使用 fallback 版本找 path
    if ep >= nrand:
        path = find_path_with_fallback(s, g, V, E)
    else:
        path = []

    sg_idx = 0
    traj = []
    refinement_triggered = False

    for t in range(T):
        s_vec = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        g_vec = torch.tensor(g, dtype=torch.float32).unsqueeze(0)

        epsilon = max(0.2 * (1 - ep / nepi), 0.05)
        if ep < nrand or sg_idx >= len(path) or random.random() < epsilon:
            a = env.action_space.sample()
        else:
            a = policy.act(s_vec, g_vec).detach().numpy().squeeze()


        obs_next, reward, terminated, truncated, info = env.step(a)
        s_next = phi(obs_next)
        traj.append((obs, g, a, reward, obs_next, terminated or truncated))

        if sg_idx < len(path) and distance(s_next, path[sg_idx]) < 0.5:
            sg_idx += 1

        if t > rt and sg_idx < len(path):
            v_fail = path[sg_idx]
            subgoal_failures[v_fail] += 1
            if subgoal_failures[v_fail] > tau_n:
                V.discard(v_fail)
                E = {(u, v) for (u, v) in E if u != v_fail and v != v_fail}
                refinement_triggered = True
            break

        obs = obs_next
        if terminated or truncated:
            break

    # HER: 從尾端選擇 8 個內的 transition
her_indices = sorted(random.sample(range(len(traj)//2, len(traj)), min(8, len(traj)//2)))

for i in her_indices:
    s_exp, g_old, a, r, s_next, d = traj[i]
    th = random.randint(1, h_max)
    future_idx = min(i + th, len(traj) - 1)
    phi_future = phi(traj[future_idx][0])

    # 從 grid 中挑最接近 phi_future 的幾個點作為 g_her 候選
    V_sorted = sorted(V, key=lambda v: np.linalg.norm(np.array(v) - phi_future))
    g_her = random.choice(V_sorted[:10])

    # 使用溫和的 reward 定義與成功判定條件
    done_her = distance(phi(s_exp), g_her) < 0.5
    r_her = np.exp(-distance(phi(s_exp), g_her))

    buffer.push(s_exp, g_her, a, r_her, s_next, float(done_her))
    if done_her:
        print(f"HER success: s={phi(s_exp)} → g_her={g_her}")

    # refine 條件判斷與執行
    if ((ep % nref == 0 and ep > 0) or refinement_triggered or should_refine(subgoal_failures, V)) and len(V) < max_V:
        V, E = refine_grid(V, E, delta0)
        for v in V:
            if v not in subgoal_failures:
                subgoal_failures[v] = 0
        print(f"[Ep {ep}] Grid refined: |V|={len(V)}")
    else:
        if (ep % nref == 0 and ep > 0) or refinement_triggered:
            print(f"[Ep {ep}] Grid too large or not enough failures, skipping refinement.")

    train_policy()

    # 成功率統計 + Checkpoint 儲存
    success = 1 if sg_idx == len(path) and len(path) > 0 else 0
    success_rates.append(success)

    if ep % 50 == 0:
        torch.save(policy.state_dict(), f"checkpoints/policy_ep{ep}.pt")

        # 成功率曲線
        plt.figure()
        plt.plot(np.convolve(success_rates, np.ones(10)/10, mode='valid'))
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (Moving Avg)")
        plt.title("BEAG Ant-v4 Success Rate")
        plt.savefig(f"plots/success_rate_ep{ep}.png")
        plt.close()

        # Grid fail heatmap
        x, y, fail = zip(*[(v[0], v[1], subgoal_failures[v]) for v in V])
        plt.figure()
        plt.scatter(x, y, c=fail, cmap='hot', s=15)
        plt.colorbar(label="Failure Count")
        plt.title(f"Subgoal Failures (ep {ep})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"plots/grid_failures_ep{ep}.png")
        plt.close()


    print(f"[Ep {ep}] Reached {sg_idx}/{len(path)} subgoals | Buffer: {len(buffer)} | V: {len(V)} | Success: {success}")

env.close()
