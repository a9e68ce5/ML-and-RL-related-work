import numpy as np
import itertools
import math
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

#############################################
# 輔助函式與常量
#############################################

def log_gamma(x, gamma):
    return torch.log(x) / math.log(gamma)

def quantize_node(node, decimals=4):
    return tuple(round(float(x), decimals) for x in node)

def compute_path_length(path):
    if path is None or len(path) < 2:
        return float('inf')
    length = 0
    for i in range(len(path) - 1):
        length += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]), ord=2)
    return length

def collision_check(env, u, v, steps=10):
    u = np.array(u, dtype=np.float32)
    v = np.array(v, dtype=np.float32)
    for t in np.linspace(0, 1, steps):
        pos = (1-t)*u + t*v
        rr = int(round(pos[0]))
        cc = int(round(pos[1]))
        if env.is_wall(rr, cc):
            return True
    return False

#############################################
# 迷宮定義：三個 U 形迷宮
#############################################

# U Maze Easy (6x6)
MAZE_U_EASY = np.array([
    [0,0,0,1,1,1],
    [0,1,0,0,0,1],
    [0,1,0,0,0,1],
    [0,1,1,1,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0],
], dtype=np.int32)

# U Maze Moderate (8x8)
MAZE_U_MODERATE = np.array([
    [0,0,0,0,1,1,1,1],
    [0,1,1,0,0,0,0,1],
    [0,1,0,0,0,1,0,1],
    [0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1],
    [0,1,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0],
], dtype=np.int32)

# U Maze Hard (10x10)
MAZE_U_HARD = np.array([
    [0,0,0,0,0,1,1,1,1,1],
    [0,1,1,1,0,0,0,0,0,1],
    [0,1,0,1,0,1,1,1,0,1],
    [0,1,0,1,0,1,1,1,0,1],
    [0,1,0,1,0,1,1,1,0,1],
    [0,1,0,1,0,0,0,0,0,1],
    [0,1,0,1,1,1,1,1,0,1],
    [0,1,0,0,0,0,0,1,0,1],
    [0,1,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0],
], dtype=np.int32)

def make_u_maze_easy():
    start = (4,1)
    goal  = (1,4)
    return MAZE_U_EASY, start, goal

def make_u_maze_moderate():
    start = (6,0)
    goal  = (1,5)
    return MAZE_U_MODERATE, start, goal

def make_u_maze_hard():
    start = (9,0)
    goal  = (2,8)
    return MAZE_U_HARD, start, goal

#############################################
# MazeEnv 定義
#############################################
class MazeEnv:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.state = self.start.copy()
        self.max_row, self.max_col = maze.shape

    def reset(self):
        self.state = self.start.copy()
        return self.state

    def step(self, action):
        steps = 50  # 插值次數，可根據需要調整
        safe = True
        next_state = self.state + np.array(action, dtype=np.float32)
        for t in np.linspace(0, 1, steps):
            pos = (1-t)*self.state + t*next_state
            rr = int(round(pos[0]))
            cc = int(round(pos[1]))
            if self.is_wall(rr, cc):
                safe = False
                break
        if not safe:
            next_state = self.state
        self.state = next_state
        done = np.linalg.norm(self.state - self.goal, ord=np.inf) < 0.5
        reward = 0 if done else -1
        return self.state.copy(), reward, done, {}

    def is_wall(self, r, c):
        if r < 0 or r >= self.max_row or c < 0 or c >= self.max_col:
            return True
        return (self.maze[r, c] == 1)

#############################################
# ValueNetwork 定義
#############################################
class ValueNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2*dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    def forward(self, u, v):
        x = torch.cat([u, v], dim=1)
        return self.net(x)

#############################################
# PolicyNetwork 定義 (最後一層加入 tanh 限制動作)
#############################################
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, max_action=0.2):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128,hidden_dim ),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        self.max_action = max_action

    def forward(self, x):
        raw_out = self.net(x)
        mean = torch.tanh(raw_out) * self.max_action  # 限制動作在 [-max_action, max_action]
        std = torch.exp(self.log_std)
        return mean, std

def sample_action(policy_net, state, subgoal):
    subgoal = quantize_node(subgoal)
    x = np.concatenate([state, np.array(subgoal)])
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    mean, std = policy_net(x)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=1)
    return action.squeeze(0).detach().numpy(), log_prob.squeeze(0)

def update_policy(policy_net, optimizer, transitions, gamma=0.99):
    R = 0
    returns = []
    for (_, _, _, r) in reversed(transitions):
        R = r + gamma*R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std()+1e-6)
    loss = 0
    for (_, _, log_prob, _), R in zip(transitions, returns):
        loss -= log_prob * R
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

#############################################
# train_value_network_with_negative_samples 定義
#############################################
def train_value_network_with_negative_samples(value_net, optimizer, transitions, gamma=0.99, negative_weight=2.0):
    criterion = nn.MSELoss(reduction='none')
    optimizer.zero_grad()
    u_batch = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
    goal_batch = torch.tensor([t[1] for t in transitions], dtype=torch.float32)
    reward_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1)
    next_u_batch = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    value_pred = value_net(u_batch, goal_batch)
    with torch.no_grad():
        value_next = value_net(next_u_batch, goal_batch)
    target = reward_batch + gamma*value_next
    loss_all = nn.MSELoss(reduction='none')(value_pred, target)
    negative_mask = (reward_batch < -1).float()
    weight = 1.0 + (negative_weight-1.0)*negative_mask
    loss = (loss_all*weight).mean()
    loss.backward()
    optimizer.step()
    return loss.item()

#############################################
# compute_edge_weight 定義 (含碰撞檢查)
#############################################
def compute_edge_weight(u, v, value_net, gamma=0.99, env=None):
    if env is not None:
        if collision_check(env, u, v, steps=50):
            return float('inf')
    u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
    v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        val = value_net(u_tensor, v_tensor)
    inner = 1.0 + (1-gamma)*val
    d_tilde = -log_gamma(inner, gamma)
    return d_tilde.item()

#############################################
# GridGraph 定義
#############################################
class GridGraph:
    def __init__(self, delta0, K, lower_bound, upper_bound, value_net, gamma=0.99, tau_n=3, env=None):
        self.delta0 = delta0
        self.K = K
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value_net = value_net
        self.gamma = gamma
        self.tau_n = tau_n
        self.env = env
        self.nodes = []
        self.edges = {}
        self.n_success = {}
        self.n_attempts = {}
        self.node_delta = {}
        self.construct_graph()

    def construct_graph(self):
        ranges = [np.arange(self.lower_bound, self.upper_bound, self.delta0) for _ in range(self.K)]
        raw_nodes = list(itertools.product(*ranges))
        self.nodes = [quantize_node(node) for node in raw_nodes]
        for node in self.nodes:
            self.n_success[node] = 0
            self.n_attempts[node] = 0
            self.node_delta[node] = self.delta0
        self.edges = {}
        for u in self.nodes:
            for v in self.nodes:
                if u != v and max(abs(a-b) for a, b in zip(u, v)) == self.delta0:
                    self.edges[(u, v)] = self.compute_weight(u, v)

    def compute_weight(self, u, v):
        weight = compute_edge_weight(u, v, self.value_net, self.gamma, env=self.env)
        if self.n_success.get(v, 0) == 0 and self.n_attempts.get(v, 0) > self.tau_n:
            weight = float('inf')
        return weight

    def remove_subgoal(self, v):
        for edge in list(self.edges.keys()):
            if v in edge:
                self.edges[edge] = float('inf')

    def adaptive_refinement(self, v):
        v = quantize_node(v)
        if v not in self.node_delta:
            self.node_delta[v] = self.delta0
        current_delta = self.node_delta[v]
        new_delta = current_delta / 2.0
        candidates = []
        for coord in v:
            candidates.append([coord - current_delta, coord - new_delta, coord, coord + new_delta, coord + current_delta])
        raw_refined_nodes = list(itertools.product(*candidates))
        refined_nodes = [quantize_node(node) for node in raw_refined_nodes]
        for node in refined_nodes:
            if node not in self.nodes:
                self.nodes.append(node)
                self.n_success[node] = 0
                self.n_attempts[node] = 0
                self.node_delta[node] = new_delta
        for u in refined_nodes:
            for w in refined_nodes:
                if u != w and max(abs(a-b) for a, b in zip(u, w)) == new_delta:
                    self.edges[(u, w)] = self.compute_weight(u, w)
        print(f"Adaptive refinement around node {v} with new interval {new_delta}")

    def find_path(self, start, goal):
        start = quantize_node(start)
        goal  = quantize_node(goal)
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
        point = quantize_node(point)
        return min(self.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(point), ord=np.inf))

#############################################
# BEAGAgent 定義
#############################################
class BEAGAgent:
    def __init__(self, env, grid_graph, policy_net, policy_optimizer):
        self.env = env
        self.grid_graph = grid_graph
        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.current_path = None
        self.current_subgoal_index = 0
        self.ttr = 0

    def feature(self, state):
        return quantize_node(state)

    def subgoal_reached(self, state, subgoal, tol=0.5):
        return np.linalg.norm(np.array(state) - np.array(subgoal), ord=np.inf) < tol

    def plan_path(self, state, goal):
        start_feature = self.feature(state)
        goal_feature  = quantize_node(goal)
        # 直接以原點作為起點
        path = self.grid_graph.find_path(start_feature, goal_feature)
        if path is not None and path[0] != start_feature:
            # 強制將起點插入 (僅供視覺化參考)
            path = (start_feature,) + tuple(path)
        return path

    def train(self, num_episodes=1000, T=50, tau_t=20, gamma=0.99):
        all_trajectories = []
        best_path = None
        best_trajectory = None
        best_length = float('inf')
        best_episode = -1

        for ep in range(num_episodes):
            state = self.env.reset()
            trajectory = [state.copy()]
            policy_transitions = []
            print(f"Episode {ep} start, initial state = {state}")
            done = False
            t = 0
            goal = self.env.goal
            self.current_path = self.plan_path(state, goal)
            if self.current_path is None:
                print("No path found!")
                all_trajectories.append((trajectory, None))
                continue
            print("Planned path (via GridGraph/Dijkstra):")
            print(self.current_path)
            self.current_subgoal_index = 0
            self.ttr = 0
            while not done and t < T:
                if self.current_subgoal_index >= len(self.current_path):
                    print("Planned path ended but goal not reached!")
                    break
                current_subgoal = quantize_node(self.current_path[self.current_subgoal_index])
                action, log_prob = sample_action(self.policy_net, state, current_subgoal)
                next_state, reward, done, _ = self.env.step(action)
                print(f"t={t}, state={state}, subgoal={current_subgoal}, action={action}, next_state={next_state}")
                policy_transitions.append((state, current_subgoal, log_prob, reward))
                if self.subgoal_reached(next_state, current_subgoal):
                    print(f"Subgoal {current_subgoal} reached")
                    if current_subgoal not in self.grid_graph.n_success:
                        self.grid_graph.n_success[current_subgoal] = 0
                    self.grid_graph.n_success[current_subgoal] += 1
                    self.current_subgoal_index = min(self.current_subgoal_index+1, len(self.current_path)-1)
                    self.ttr = 0
                else:
                    if current_subgoal not in self.grid_graph.n_attempts:
                        self.grid_graph.n_attempts[current_subgoal] = 0
                    self.grid_graph.n_attempts[current_subgoal] += 1
                    self.ttr += 1
                    if self.ttr > tau_t:
                        print(f"Subgoal {current_subgoal} repeatedly failed, performing refinement.")
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
            ploss = update_policy(self.policy_net, self.policy_optimizer, policy_transitions, gamma)
            print(f"Episode {ep} policy loss = {ploss:.4f}")
            print(f"Episode {ep} finished.\n")
            all_trajectories.append((trajectory, self.current_path))
            if done and self.current_path is not None:
                path_len = compute_path_length(self.current_path)
                if path_len < best_length:
                    best_length = path_len
                    best_path = self.current_path
                    best_trajectory = trajectory
                    best_episode = ep

        print(f"Best path found in episode {best_episode}, length = {best_length:.4f}")
        return best_path, best_trajectory, all_trajectories

#############################################
# 視覺化函式 (加入網格線、網格節點顏色區分 adaptive grid 更新，及 maze_name)
#############################################
def visualize_trajectory(trajectory, grid_graph, planned_path, bfs_path, start, goal, maze_env=None, maze_name=""):
    plt.figure(figsize=(12,12))
    traj = np.array(trajectory)
    plt.plot(traj[:,1], traj[:,0], 'b.-', label="Agent Trajectory")
    if planned_path is not None:
        path_arr = np.array(planned_path)
        plt.plot(path_arr[:,1], path_arr[:,0], 'ro-', label="Planned Path", markersize=8, linewidth=2)
    if bfs_path is not None:
        bfs_arr = np.array(bfs_path)
        plt.plot(bfs_arr[:,1], bfs_arr[:,0], 'g.-', label="BFS Shortest Path", markersize=8, linewidth=2)
    plt.scatter(start[1], start[0], c='green', s=150, marker='*', label="Start")
    plt.scatter(goal[1], goal[0], c='red', s=150, marker='*', label="Goal")
    
    if maze_env is not None:
        ax = plt.gca()
        # 畫牆壁
        for r in range(maze_env.max_row):
            for c in range(maze_env.max_col):
                if maze_env.maze[r, c] == 1:
                    rect = plt.Rectangle((c, r), 1, 1, facecolor='black', alpha=0.5)
                    ax.add_patch(rect)
        # 畫出網格節點，根據是否 refine 分為兩種顏色
        original_nodes = []
        refined_nodes = []
        for node in grid_graph.nodes:
            if grid_graph.node_delta[node] == grid_graph.delta0:
                original_nodes.append(node)
            else:
                refined_nodes.append(node)
        if original_nodes:
            original_nodes = np.array(original_nodes)
            plt.scatter(original_nodes[:,1], original_nodes[:,0], c='gray', s=10, alpha=0.3, label="Original Grid Nodes")
        if refined_nodes:
            refined_nodes = np.array(refined_nodes)
            plt.scatter(refined_nodes[:,1], refined_nodes[:,0], c='orange', s=10, alpha=0.5, label="Refined Grid Nodes")
        # 畫出網格線 (以原始 delta0 為間隔)
        delta = grid_graph.delta0
        for x in np.arange(0, maze_env.max_col+1, delta):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
        for y in np.arange(0, maze_env.max_row+1, delta):
            plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
        plt.xlim(0, maze_env.max_col)
        plt.ylim(maze_env.max_row, 0)
    plt.title(f"Comparison: BFS vs Agent - Maze: {maze_name}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

#############################################
# BFS: 直接在迷宮上找最短路徑 (僅供比較)
#############################################
def bfs_shortest_path(maze, start, goal):
    max_r, max_c = maze.shape
    if maze[start[0], start[1]] == 1 or maze[goal[0], goal[1]] == 1:
        return None
    visited = set()
    queue = deque()
    queue.append((start, [start]))
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for d in directions:
            nr = current[0] + d[0]
            nc = current[1] + d[1]
            if 0 <= nr < max_r and 0 <= nc < max_c and maze[nr, nc] == 0:
                queue.append(((nr, nc), path+[(nr, nc)]))
    return None

#############################################
# 主程式
#############################################
if __name__ == "__main__":
    # 顯示所有迷宮 (可選)
    def visualize_all_mazes():
        mazes = [
            ("U Maze Easy", MAZE_U_EASY, *make_u_maze_easy()[1:]),
            ("U Maze Moderate", MAZE_U_MODERATE, *make_u_maze_moderate()[1:]),
            ("U Maze Hard", MAZE_U_HARD, *make_u_maze_hard()[1:])
        ]
        plt.figure(figsize=(15,5))
        for i, (name, maze, start, goal) in enumerate(mazes):
            plt.subplot(1,3,i+1)
            plt.imshow(maze, cmap='gray_r')
            plt.title(name)
            plt.scatter(start[1], start[0], c='green', s=100, label="Start")
            plt.scatter(goal[1], goal[0], c='red', s=100, label="Goal")
            plt.legend()
            plt.axis("equal")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    visualize_all_mazes()

    # 選擇迷宮： "u_maze_easy", "u_maze_moderate", "u_maze_hard"
    env_name = "u_maze_hard"
    if env_name == "u_maze_easy":
        maze_array, start, goal = make_u_maze_easy()
    elif env_name == "u_maze_moderate":
        maze_array, start, goal = make_u_maze_moderate()
    elif env_name == "u_maze_hard":
        maze_array, start, goal = make_u_maze_hard()
    else:
        print("Unknown maze; using U Maze Easy.")
        maze_array, start, goal = make_u_maze_easy()

    env = MazeEnv(maze_array, start, goal)

    # BFS 最短路徑 (僅供比較)
    bfs_path = bfs_shortest_path(maze_array, start, goal)
    if bfs_path is not None:
        print("BFS shortest path found:")
        print(bfs_path)
        print("BFS path length =", compute_path_length(bfs_path))
    else:
        print("BFS did not find a path.")

    # 建立 ValueNetwork 與 PolicyNetwork
    dim = 2
    value_net = ValueNetwork(dim, hidden_dim=64)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
    gamma = 0.99

    # 預訓練 ValueNetwork 用 dummy data
    dummy_transitions = [
        (np.array([1.0,1.0]), np.array([4.0,4.0]), -1, np.array([1.5,1.5])),
        (np.array([1.5,1.5]), np.array([4.0,4.0]), -1, np.array([2.0,2.0])),
        (np.array([2.0,2.0]), np.array([4.0,4.0]), -1, np.array([2.5,2.5])),
        (np.array([2.5,2.5]), np.array([4.0,4.0]),  0, np.array([4.0,4.0]))
    ]
    for i in range(100):
        loss = train_value_network_with_negative_samples(value_net, value_optimizer, dummy_transitions, gamma, negative_weight=2.0)
        if i % 10 == 0:
            print(f"[ValueNet pretrain] iteration {i}, loss = {loss:.4f}")
    value_net.eval()

    policy_net = PolicyNetwork(input_dim=4, output_dim=2, hidden_dim=64, max_action=0.5)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # 建立 GridGraph (用於路徑規劃)
    grid_graph = GridGraph(
        delta0 = 0.5,
        K = 2,
        lower_bound = 0.0,
        upper_bound = float(max(env.max_row, env.max_col)),
        value_net = value_net,
        gamma = gamma,
        tau_n = 4,
        env = env
    )

    # 建立 BEAGAgent 並訓練
    agent = BEAGAgent(env, grid_graph, policy_net, policy_optimizer)
    best_path, best_trajectory, trajectories = agent.train(num_episodes=3000, T=300, tau_t=60, gamma=gamma)

    print("\nBest planned path obtained:")
    if best_path is not None:
        for node in best_path:
            rr, cc = int(round(node[0])), int(round(node[1]))
            wall = env.is_wall(rr, cc)
            print(f"  Node: {node}, Rounded: ({rr}, {cc}), Is wall: {wall}")
    else:
        print("No successful path found.")

    if best_path is not None and best_trajectory is not None:
        visualize_trajectory(best_trajectory, grid_graph, best_path, bfs_path, env.start, env.goal, maze_env=env, maze_name=env_name)
    else:
        print("No successful path found.")
