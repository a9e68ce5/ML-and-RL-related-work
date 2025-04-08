
# MDP Scenario Analysis

## Experiment Setup

### Overview
本實驗評估並比較三個不同的馬可夫決策過程（MDP）場景，目的是觀察在不同的獎勵（Rewards）、終端狀態（Terminal States）和隨機性（Noise）改變下，最佳策略（Optimal Policy）與狀態值（State Values）的變化。

### Experiment Methods
- **Value Iteration**：逐步更新每個狀態的值（Values），直到所有狀態值收斂到最佳值函數。
- **Policy Iteration**：依序執行「策略評估（Policy Evaluation）」和「策略改進（Policy Improvement）」步驟，直到策略穩定為最佳策略。

---

## **Scenario 0: 原始實驗設定**

### **設定**
- **Grid Size**: 4x4
- **Noise**: `0.1`（10% 機率執行垂直於預期行動的移動）
- **Terminal States**: `(0, 0)`, `(0, 1)`, `(1, 1)`
- **Rewards**:
  ```
   +1    -1     0     0
    0    -1     0     0
    0     0     0     0
    0     0     0     0
  ```

---

### **結果**

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

## **Scenario 1: 增強獎勵**

### **設定變更**
- 與原始實驗相比，此場景具有以下差異：
  - **Rewards**:
    - `(0, 0)` 的獎勵從 `+1` 增加至 `+10`。
    - `(1, 3)` 增加一個新的獎勵 `+5`。
  - **Terminal States**:
    - 僅保留 `(0, 0)` 作為唯一的終端狀態。
  - **Noise**: 保持不變，仍為 `0.1`。

---

### **結果**

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

---

## **Scenario 2: 增加隨機性與新終端狀態**

### **設定變更**
- 與原始實驗相比，此場景具有以下差異：
  - **Noise**: 從 `0.1` 增加至 `0.4`，隨機性大幅提升。
  - **Terminal States**:
    - 新增 `(2, 0)` 作為額外的終端狀態。
  - **Rewards**:
    - `(3, 3)` 增加一個新的正獎勵 `+2`。
    - `(0, 1)` 和 `(1, 1)` 保持負獎勵 `-1`。

---

### **結果**

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
  - 在原始場景中，代理人會避免負獎勵並嘗試平衡移動至 `(0, 0)`。

- **Scenario 1**: 
  - 代理人因高獎勵值的影響，策略集中於快速移動至 `(0, 0)`。

- **Scenario 2**: 
  - 增加的隨機性與多個終端狀態導致策略更加謹慎，代理人在風險與報酬之間進行平衡。

---

此 `README.md` 詳細記錄了每個場景的設定變更、結果與策略分析。
