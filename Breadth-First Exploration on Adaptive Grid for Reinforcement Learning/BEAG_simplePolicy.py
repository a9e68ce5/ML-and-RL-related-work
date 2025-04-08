
import numpy as np
import itertools
import math
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt

# ------------------------------
# Ṽπ 神經網路實作 (全連接網路)
# ------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        """
        dim: 目標空間的維度，u 與 v 為 dim 維向量，輸入層大小為 2*dim。
        """
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 輸出非負
        )
        
    def forward(self, u, v):
        """
        u, v: tensor，形狀均為 (batch_size, dim)
        返回: (batch_size, 1) 的值估計
        """
        x = torch.cat([u, v], dim=1)
        return self.net(x)

def log_gamma(x, gamma):
    """計算 log_γ(x) = log(x) / log(γ)"""
    return torch.log(x) / math.log(gamma)

# ------------------------------
# 用於訓練 ValueNetwork 的函數（加入負樣本加權）
# ------------------------------
def train_value_network_with_negative_samples(value_net, optimizer, transitions, gamma=0.99, negative_weight=2.0):
    """
    transitions: 每個元素為 (u, goal, reward, next_u)
      u: 當前狀態或子目標 (numpy array)
      goal: 目標 (numpy array)
      reward: 獎勵（例如 -1 為正常，-10 為碰撞牆壁的負樣本）
      next_u: 下個狀態表示
    negative_weight: 負樣本損失權重倍數
    使用 TD 目標更新：target = reward + gamma * Ṽπ(next_u, goal)
    """
    criterion = nn.MSELoss(reduction='none')
    optimizer.zero_grad()
    
    u_batch = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
    goal_batch = torch.tensor(np.array([t[1] for t in transitions]), dtype=torch.float32)
    reward_batch = torch.tensor(np.array([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(1)
    next_u_batch = torch.tensor(np.array([t[3] for t in transitions]), dtype=torch.float32)
    
    value_pred = value_net(u_batch, goal_batch)
    with torch.no_grad():
        value_next = value_net(next_u_batch, goal_batch)
    target = reward_batch + gamma * value_next
    loss_all = criterion(value_pred, target)
    
    # 對於 reward 小於 -1 的視為負樣本，給予較大權重
    negative_mask = (reward_batch < -1).float()
    weight = 1.0 + (negative_weight - 1.0) * negative_mask
    loss = (loss_all * weight).mean()
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# ------------------------------
# 線段插值碰撞檢查函數
# ------------------------------
def collision_check(env, u, v, steps=15):
    """
    env: MazeEnv，需提供 is_wall(row, col) 方法
    u, v: (row, col) 形式的座標 (tuple 或 numpy array)
    steps: 插值步數
    回傳: True 表示 u->v 之間有碰牆，False 表示無碰牆
    """
    u = np.array(u, dtype=np.float32)
    v = np.array(v, dtype=np.float32)
    for t in np.linspace(0, 1, steps):
        pos = (1 - t) * u + t * v
        row, col = int(round(pos[0])), int(round(pos[1]))
        if env.is_wall(row, col):
            return True
    return False

# ------------------------------
# 利用 ValueNetwork 計算邊權重，加入碰撞檢查
# ------------------------------
def compute_edge_weight(u, v, value_net, gamma=0.99, env=None):
    """
    u, v: 目標空間中的節點 (tuple, 1D)
    若 env 提供，先做線段插值碰撞檢查，若碰牆則回傳 ∞
    否則計算 w(u,v) = - log_γ(1+(1-γ)*Ṽπ(u|v))
    """
    if env is not None:
        if collision_check(env, u, v, steps=15):
            return float('inf')
    u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
    v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        val = value_net(u_tensor, v_tensor)
    inner = 1.0 + (1 - gamma) * val
    d_tilde = -log_gamma(inner, gamma)  # 負號使權重為正
    return d_tilde.item()

# ------------------------------
# GridGraph：網格圖構建與管理
# ------------------------------
class GridGraph:
    def __init__(self, delta0, K, lower_bound, upper_bound, value_net, gamma=0.99, tau_n=3, env=None):
        """
        delta0: 初始網格間隔
        K: 目標空間維度
        lower_bound, upper_bound: 目標空間上下界
        value_net: 已訓練好的 ValueNetwork，用以計算邊權重
        tau_n: 失敗次數門檻
        env: MazeEnv 或類似環境，用於碰撞檢查
        """
        self.delta0 = delta0
        self.K = K
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value_net = value_net
        self.gamma = gamma
        self.tau_n = tau_n
        self.env = env  # 傳入環境對象
        self.nodes = []
        self.edges = {}
        self.n_success = {}
        self.n_attempts = {}
        self.node_delta = {}
        self.construct_graph()

    def construct_graph(self):
        ranges = [np.arange(self.lower_bound, self.upper_bound, self.delta0) for _ in range(self.K)]
        self.nodes = list(itertools.product(*ranges))
        for node in self.nodes:
            self.n_success[node] = 0
            self.n_attempts[node] = 0
            self.node_delta[node] = self.delta0
        self.edges = {}
        for u in self.nodes:
            for v in self.nodes:
                if u != v and max([abs(a - b) for a, b in zip(u, v)]) == self.delta0:
                    self.edges[(u, v)] = self.compute_weight(u, v)

    def compute_weight(self, u, v):
        weight = compute_edge_weight(u, v, self.value_net, self.gamma, env=self.env)
        if self.n_success[v] == 0 and self.n_attempts[v] > self.tau_n:
            weight = float('inf')
        return weight

    def remove_subgoal(self, v):
        for edge in list(self.edges.keys()):
            if v in edge:
                self.edges[edge] = float('inf')

    def adaptive_refinement(self, v):
        current_delta = self.node_delta[v]
        new_delta = current_delta / 2.0
        candidates = []
        for coord in v:
            candidates.append([coord - current_delta, coord - new_delta, coord, coord + new_delta, coord + current_delta])
        refined_nodes = list(itertools.product(*candidates))
        for node in refined_nodes:
            if node not in self.nodes:
                self.nodes.append(node)
                self.n_success[node] = 0
                self.n_attempts[node] = 0
                self.node_delta[node] = new_delta
        for u in refined_nodes:
            for w in refined_nodes:
                if u != w and max([abs(a - b) for a, b in zip(u, w)]) == new_delta:
                    self.edges[(u, w)] = self.compute_weight(u, w)
        print(f"Adaptive refinement around node {v} with new interval {new_delta}")

    def find_path(self, start, goal):
        # 若起點或終點不在圖中，則先加入
        if start not in self.nodes:
            self.nodes.append(start)
            self.n_success[start] = 0
            self.n_attempts[start] = 0
            self.node_delta[start] = self.delta0
        if goal not in self.nodes:
            self.nodes.append(goal)
            self.n_success[goal] = 0
            self.n_attempts[goal] = 0
            self.node_delta[goal] = self.delta0

        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for (u, v), weight in self.edges.items():
            if weight < float('inf'):
                G.add_edge(u, v, weight=weight)
        try:
            path = nx.dijkstra_path(G, start, goal, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None

    def nearest_node(self, point):
        """返回與 point 在 L∞ 距離下最近的節點"""
        return min(self.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(point), ord=np.inf))

# ------------------------------
# MazeEnv：迷宮型環境，模擬類似 AntMaze 的情境
# ------------------------------
class MazeEnv:
    def __init__(self, maze, start, goal):
        """
        maze: 2D numpy array，0 表示通道，1 表示牆壁
        start: 起始位置 (row, col)
        goal: 目標位置 (row, col)
        """
        self.maze = maze
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.state = self.start.copy()
        self.max_row, self.max_col = maze.shape

    def reset(self):
        self.state = self.start.copy()
        return self.state

    def step(self, action):
        """
        action: [dx, dy]，假設每次移動 1 單位（可根據需要調整）
        """
        next_state = self.state + np.array(action, dtype=np.float32)
        # 四捨五入取整數作為格子索引
        row, col = int(round(next_state[0])), int(round(next_state[1]))
        # 越界檢查
        if row < 0 or row >= self.max_row or col < 0 or col >= self.max_col:
            next_state = self.state
        # 牆壁檢查
        elif self.maze[row, col] == 1:
            next_state = self.state
        self.state = next_state
        done = np.linalg.norm(self.state - self.goal, ord=np.inf) < 0.5
        reward = 0 if done else -1
        return self.state, reward, done, {}

    def is_wall(self, row, col):
        """檢查 (row, col) 是否為牆壁或越界"""
        if row < 0 or row >= self.max_row or col < 0 or col >= self.max_col:
            return True
        return (self.maze[row, col] == 1)

# ------------------------------
# 固定環境起始狀態與目標設定（若環境支援）
# ------------------------------
def set_fixed_goal_env(env, start, goal):
    env.fixed_start = np.array(start, dtype=np.float32)
    env.fixed_goal = np.array(goal, dtype=np.float32)
    return env

# ------------------------------
# BEAGAgent：利用網格圖進行子目標規劃與訓練
# ------------------------------
class BEAGAgent:
    def __init__(self, env, grid_graph, policy):
        self.env = env
        self.grid_graph = grid_graph
        self.policy = policy
        self.current_path = None
        self.current_subgoal_index = 0
        self.ttr = 0  # 當前子目標嘗試步數

    def feature(self, state):
        # 將狀態轉換為 tuple，每個元素轉為 python float
        return tuple(float(x) for x in state)

    def subgoal_reached(self, state, subgoal, tol=0.5):
        return np.linalg.norm(np.array(state) - np.array(subgoal), ord=np.inf) < tol

    def plan_path(self, state, goal):
        # 將狀態與目標映射到網格上：若不在圖中則取最近節點
        start_feature = self.feature(state)
        goal_feature = tuple(float(x) for x in goal)
        start_node = self.grid_graph.nearest_node(start_feature)
        goal_node = self.grid_graph.nearest_node(goal_feature)
        path = self.grid_graph.find_path(start_node, goal_node)
        return path

    def train(self, num_episodes=10, T=50, tau_t=10):
        all_trajectories = []  # 儲存每個 episode 的軌跡及規劃路徑
        for ep in range(num_episodes):
            state = self.env.reset()
            trajectory = [state.copy()]
            print(f"Episode {ep} start, initial state: {state}")
            done = False
            t = 0
            goal = self.env.goal
            self.current_path = self.plan_path(state, goal)
            if self.current_path is None:
                print("No path found!")
                break
            # 列印規劃路徑座標以驗證是否避開牆壁
            print("Planned path coordinates:")
            for node in self.current_path:
                row, col = int(round(node[0])), int(round(node[1]))
                is_wall = self.env.is_wall(row, col)
                print(f"  Node: {node}, Rounded: ({row}, {col}), Is wall: {is_wall}")
            self.current_subgoal_index = 0
            self.ttr = 0
            while not done and t < T:
                if self.current_subgoal_index >= len(self.current_path):
                    print("Planned path ended but goal not reached!")
                    break
                current_subgoal = self.current_path[self.current_subgoal_index]
                action = self.policy(state, current_subgoal)
                next_state, reward, done, _ = self.env.step(action)
                print(f"t={t}, state={state}, subgoal={current_subgoal}, action={action}, next_state={next_state}")
                if self.subgoal_reached(next_state, current_subgoal):
                    print(f"Subgoal {current_subgoal} reached")
                    self.grid_graph.n_success[current_subgoal] += 1
                    self.current_subgoal_index = min(self.current_subgoal_index + 1, len(self.current_path) - 1)
                    self.ttr = 0
                else:
                    self.grid_graph.n_attempts[current_subgoal] += 1
                    self.ttr += 1
                    if self.ttr > tau_t:
                        print(f"Subgoal {current_subgoal} repeatedly failed, performing adaptive refinement")
                        self.grid_graph.remove_subgoal(current_subgoal)
                        self.grid_graph.adaptive_refinement(current_subgoal)
                        self.current_path = self.plan_path(state, goal)
                        self.current_subgoal_index = 0
                        self.ttr = 0
                        if self.current_path is None:
                            print("After refinement, no path found!")
                            break
                state = next_state
                trajectory.append(state.copy())
                t += 1
            print(f"Episode {ep} finished.\n")
            all_trajectories.append((trajectory, self.current_path))
        return all_trajectories

# ------------------------------
# 簡單策略：根據當前狀態朝向子目標前進 (固定步長)
# ------------------------------
def simple_policy(state, subgoal, step_size=0.5):
    state_arr = np.array(state)
    subgoal_arr = np.array(subgoal)
    direction = subgoal_arr - state_arr
    norm = np.linalg.norm(direction) + 1e-6
    return (step_size * direction / norm).tolist()

# ------------------------------
# 視覺化軌跡與規劃路徑
# ------------------------------
def visualize_trajectory(trajectory, grid_graph, planned_path, start, goal, maze_env=None):
    """
    trajectory: 代理狀態軌跡 (list of [x, y])
    grid_graph: GridGraph 物件
    planned_path: 規劃出的子目標路徑 (list of tuple)
    start: 起點 [x, y]
    goal: 目標 [x, y]
    maze_env: (選填) 如果有 MazeEnv，則使用 maze 屬性顯示牆壁
    """
    traj = np.array(trajectory)
    
    plt.figure(figsize=(8, 8))
    # 如果提供 maze_env，顯示迷宮背景
    if maze_env is not None:
        plt.imshow(maze_env.maze, cmap='gray_r', origin='upper',
                   extent=[0, maze_env.max_col, maze_env.max_row, 0])
    
    # 畫出網格節點（假設節點以 (row, col) 表示）
    nodes = np.array(grid_graph.nodes)
    plt.scatter(nodes[:, 1], nodes[:, 0], c='gray', s=10, alpha=0.5, label="Grid Nodes")
    
    # 畫出規劃路徑
    if planned_path is not None:
        path_arr = np.array(planned_path)
        plt.plot(path_arr[:, 1], path_arr[:, 0], 'ro-', label="Planned Path", markersize=8, linewidth=2)
    
    # 畫出代理軌跡
    plt.plot(traj[:, 1], traj[:, 0], 'b.-', label="Agent Trajectory")
    
    # 畫出起點與目標
    plt.scatter(start[1], start[0], c='green', s=100, label="Start")
    plt.scatter(goal[1], goal[0], c='red', s=100, label="Goal")
    
    # 調整繪圖範圍與保持等比例
    plt.xlim(-1, maze_env.max_col + 1)
    plt.ylim(maze_env.max_row + 1, -1)
    plt.axis('equal')
    
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title("Agent Trajectory, Planned Path and Maze")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------
# MazeEnv：迷宮型環境，模擬類似 AntMaze 的情境
# ------------------------------
class MazeEnv:
    def __init__(self, maze, start, goal):
        """
        maze: 2D numpy array，0 表示通道，1 表示牆壁
        start: 起始位置 (row, col)
        goal: 目標位置 (row, col)
        """
        self.maze = maze
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.state = self.start.copy()
        self.max_row, self.max_col = maze.shape

    def reset(self):
        self.state = self.start.copy()
        return self.state

    def step(self, action):
        """
        action: [dx, dy]，假設每次移動 1 單位（可根據需要調整）
        """
        next_state = self.state + np.array(action, dtype=np.float32)
        # 四捨五入取整數作為格子索引
        row, col = int(round(next_state[0])), int(round(next_state[1]))
        # 越界檢查
        if row < 0 or row >= self.max_row or col < 0 or col >= self.max_col:
            next_state = self.state
        # 牆壁檢查
        elif self.maze[row, col] == 1:
            next_state = self.state
        self.state = next_state
        done = np.linalg.norm(self.state - self.goal, ord=np.inf) < 0.5
        reward = 0 if done else -1
        return self.state, reward, done, {}

    def is_wall(self, row, col):
        """檢查 (row, col) 是否為牆壁或越界"""
        if row < 0 or row >= self.max_row or col < 0 or col >= self.max_col:
            return True
        return (self.maze[row, col] == 1)

# ------------------------------
# 固定環境起始狀態與目標設定（若環境支援）
# ------------------------------
def set_fixed_goal_env(env, start, goal):
    env.fixed_start = np.array(start, dtype=np.float32)
    env.fixed_goal = np.array(goal, dtype=np.float32)
    return env

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    # 設定目標空間維度 (例如：AntMaze 常用 2 維)
    dim = 2
    value_net = ValueNetwork(dim, hidden_dim=64)
    optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
    gamma = 0.99

    # 構造包含負樣本的 transition 數據：
    # 正樣本（正常移動）：reward = -1
    # 負樣本（例如碰撞牆壁，狀態不變）：reward = -10000
    dummy_transitions = [
        (np.array([1.0, 1.0]), np.array([8.0, 8.0]), -1, np.array([1.5, 1.5])),
        (np.array([1.5, 1.5]), np.array([8.0, 8.0]), -1, np.array([2.0, 2.0])),
        (np.array([2.0, 2.0]), np.array([8.0, 8.0]), -1, np.array([2.5, 2.5])),
        (np.array([2.5, 2.5]), np.array([8.0, 8.0]),  0, np.array([8.0, 8.0])),
        # 負樣本：代理嘗試穿越牆壁，狀態未改變
        (np.array([3.0, 3.0]), np.array([8.0, 8.0]), -10000, np.array([3.0, 3.0]))
    ]
    for i in range(100):
        loss = train_value_network_with_negative_samples(value_net, optimizer, dummy_transitions, gamma, negative_weight=2.0)
        if i % 10 == 0:
            print(f"Training iteration {i}, loss: {loss:.4f}")
    value_net.eval()
    
    # 建立 MazeEnv 迷宮環境 (模擬 AntMaze 情境)
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    start = (0, 0)  # 迷宮左上角
    goal = (4, 4)   # 迷宮右下角
    env = MazeEnv(maze, start, goal)
    try:
        env = set_fixed_goal_env(env, start, goal)
    except Exception as e:
        pass
    
    # 建立 GridGraph 時，傳入 env 以做碰撞檢查
    delta0 = 1.0
    K = 2
    lower_bound = 0.0
    upper_bound = 10.0
    grid_graph = GridGraph(delta0, K, lower_bound, upper_bound, value_net, gamma=gamma, tau_n=4, env=env)
    
    # 建立 BEAGAgent（使用簡單策略）
    class BEAGAgentWrapper(BEAGAgent):
        pass
    agent = BEAGAgentWrapper(env, grid_graph, simple_policy)
    
    # 執行模擬／訓練，例如 1000 個 episode，每個最多 50 步
    results = agent.train(num_episodes=1000, T=50, tau_t=10)
    
    # 新增：列印規劃路徑座標以驗證是否避開牆壁
    if results and results[0][1] is not None:
        print("Planned path coordinates (rounded) and wall check:")
        for node in results[0][1]:
            row, col = int(round(node[0])), int(round(node[1]))
            wall = env.is_wall(row, col)
            print(f"Node: {node}, Rounded: ({row}, {col}), Is wall: {wall}")
    
    # 視覺化第一個 episode 的結果
    if results:
        trajectory, planned_path = results[0]
        visualize_trajectory(trajectory, grid_graph, planned_path, env.start, env.goal, maze_env=env)
