
# MDP Scenario Analysis

## Experiment Setup

### Overview
This experiment evaluates and compares three different Markov Decision Process (MDP) scenarios to observe how changes in rewards, terminal states, and noise affect optimal policies and state values.

### Experiment Methods
- **Value Iteration**: Iteratively updates state values until they converge to the optimal value function.
- **Policy Iteration**: Alternates between "Policy Evaluation" and "Policy Improvement" steps until the policy stabilizes as optimal.

---

## **Scenario 0: Original Experiment Setup**

### **Settings**
- **Grid Size**: 4x4
- **Noise**: `0.1` (10% chance of unintended perpendicular movement)
- **Terminal States**: `(0, 0)`, `(0, 1)`, `(1, 1)`
- **Rewards**:
  ```
   +1    -1     0     0
    0    -1     0     0
    0     0     0     0
    0     0     0     0
  ```

---

### **Results**

- **Values from Value Iteration**:
  ```
  0.00    0.00    0.42    0.44
  0.77    0.00    0.45    0.48
  0.71    0.59    0.55    0.51
  0.66    0.62    0.58    0.54
  ```

- **Optimal Policy Visualization**:
  ```
  .       .       >       v
  ^       .       >       v
  ^       v       <       <
  ^       <       <       <
  ```

---

## **Scenario 1: Increased Rewards**

### **Setting Changes**
- Compared to the original experiment, this scenario differs as follows:
  - **Rewards**:
    - Reward at `(0, 0)` increased from `+1` to `+10`.
    - A new reward of `+5` added at `(1, 3)`.
  - **Terminal States**:
    - Only `(0, 0)` remains as the terminal state.
  - **Noise**: Remains unchanged at `0.1`.

---

### **Results**

- **Values from Value Iteration**:
  ```
  0.00    69.11    73.58    77.94
  60.84   71.85    77.47    78.34
  63.58   68.42    73.07    77.88
  61.12   64.83    68.65    72.61
  ```

- **Optimal Policy Visualization**:
  ```
  .       >       >       v
  >       >       >       >
  >       >       >       ^
  >       >       >       ^
  ```

### **Result Explanation**
- The agent strongly prioritizes moving to `(0, 0)` due to the high reward of `+10`. 
- Regardless of starting position, the agent moves aggressively toward the terminal state.
- The reward at `(1, 3)` has limited influence, affecting the policy only when nearby.

---

## **Scenario 2: Increased Noise and New Terminal State**

### **Setting Changes**
- Compared to the original experiment, this scenario differs as follows:
  - **Noise**: Increased from `0.1` to `0.4`, significantly increasing randomness.
  - **Terminal States**:
    - A new terminal state is added at `(2, 0)`.
  - **Rewards**:
    - A new positive reward of `+2` is added at `(3, 3)`.
    - Negative rewards of `-1` remain at `(0, 1)` and `(1, 1)`.

---

### **Results**

- **Values from Value Iteration**:
  ```
  0.00    0.00    7.68    8.79
  2.96    5.35    8.14    9.94
  0.00    5.96    8.77    10.30
  6.19    7.00    8.74    10.96
  ```

- **Optimal Policy Visualization**:
  ```
  .       .       >       >
  ^       v       >       >
  .       >       >       >
  v       v       >       >
  ```

### **Result Explanation**
- The increased noise reduces predictability of agent actions. While `(0, 0)` remains a terminal state, the agent also considers moving toward `(3, 3)` for the positive reward.
- The policy demonstrates a balance between risk and reward, avoiding negative reward areas like `(0, 1)`.

---

## **Detailed Comparison**

| Feature             | Scenario 0                        | Scenario 1                       | Scenario 2                       |
|---------------------|------------------------------------|-----------------------------------|-----------------------------------|
| Grid Size           | 4x4                                | 4x4                               | 4x4                               |
| Noise               | 0.1                                | 0.1                               | 0.4                               |
| Terminal States     | `(0, 0)`, `(0, 1)`, `(1, 1)`       | `(0, 0)`                          | `(0, 0)`, `(0, 1)`, `(2, 0)`      |
| Key Rewards         | `+1` at `(0, 0)`, `-1` at `(0, 1)` | `+10` at `(0, 0)`, `+5` at `(1, 3)` | `+1` at `(0, 0)`, `+2` at `(3, 3)`, `-1` at `(0, 1)` |
| Optimal Policy      | Balanced cautious movement         | Aggressive movement to `(0, 0)`   | Balanced movement avoiding traps  |
| State Values        | Balanced values across states      | Dominated by high reward at `(0, 0)` | Distributed due to increased noise |

---

## **Discussion**

- **Scenario 0**: 
  - The agent balances risk and reward, avoiding negative rewards while moving toward `(0, 0)`.

- **Scenario 1**: 
  - The high reward at `(0, 0)` drives aggressive movement toward this terminal state.

- **Scenario 2**: 
  - Increased noise and additional terminal states lead to more cautious strategies as the agent balances risk and reward.

---

This `README.md` details each scenario's setting changes, results, and policy analysis.
