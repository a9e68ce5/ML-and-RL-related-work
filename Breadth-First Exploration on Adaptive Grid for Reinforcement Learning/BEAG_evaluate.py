# evaluate.py - 用於評估訓練後的 policy 成效
import gymnasium as gym
import numpy as np
import torch
from BEAG import Policy, phi, find_path, distance, construct_grid_graph

# --- 評估參數 ---
EPISODES = 20
GOAL = (4.0, 4.0)
THRESHOLD = 0.5
CHECKPOINT_PATH = "checkpoints/policy_ep300.pt"  # ← 可替換你要載入的模型

# --- 建立環境與 policy ---
env = gym.make("Ant-v4", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = Policy()
policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
policy.eval()

V, E = construct_grid_graph(1.0)

successes = 0

for ep in range(EPISODES):
    obs, info = env.reset()
    s = phi(obs)
    path = find_path(tuple(np.round(s)), GOAL, V, E)
    sg_idx = 0

    for t in range(500):
        s_vec = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        g_vec = torch.tensor(GOAL, dtype=torch.float32).unsqueeze(0)

        action = policy.act(s_vec, g_vec).detach().numpy().squeeze()
        obs_next, reward, terminated, truncated, _ = env.step(action)
        s_next = phi(obs_next)

        if sg_idx < len(path) and distance(s_next, path[sg_idx]) < THRESHOLD:
            sg_idx += 1

        if terminated or truncated:
            break
        obs = obs_next

    success = sg_idx == len(path) and len(path) > 0
    print(f"[Eval Ep {ep}] Reached {sg_idx}/{len(path)} subgoals | Success: {success}")
    if success:
        successes += 1

env.close()

print(f"\n✅ Evaluation Complete: {successes}/{EPISODES} success | Success Rate: {successes / EPISODES:.2f}")
