# 🧠 Machine Learning & Reinforcement Learning Portfolio

Welcome! This portfolio showcases several of my personal and academic projects across two domains:

- **Reinforcement Learning (RL)** — foundational algorithm implementations + paper-inspired simulation
- **Machine Learning (ML)** — core model implementations + applied classification project

Each project is organized in a dedicated folder with documentation, source code, and (when applicable) visual results.

---

## 🔁 Reinforcement Learning Projects (`RL`)

### 📂 `MDP`, `Q-learning`, `Policy-Gradient`

Fundamental reinforcement learning algorithms implemented from scratch for gridworld-like environments.  
Code focuses on concept clarity, custom visualization, and easy reproducibility.

---

### 📂 `BFEAG`: Breadth-First Exploration on Adaptive Grid  
Paper-inspired simulation project based on NeurIPS 2023 paper:  
**_"Breadth-First Exploration on Adaptive Grid for Reinforcement Learning"_**

#### 🎯 What I Did:
- Created discrete grid mazes of various difficulty levels
- Simulated and visualized:
  - BFS shortest path
  - Agent’s actual trajectory
  - Planned policy path
- Compared how exploration unfolds versus ideal planning

#### 🖼️ Sample Visualizations:

| Maze: Easy | Maze: Moderate | Maze: Hard |
|------------|----------------|------------|
| ![easy](./Breadth-First%20Exploration%20on%20Adaptive%20Grid%20for%20Reinforcement%20Learning/plots/Maze_easy_good.png) | ![moderate](./Breadth-First%20Exploration%20on%20Adaptive%20Grid%20for%20Reinforcement%20Learning/plots/Maze_moderate_good.png) | ![hard](./Breadth-First%20Exploration%20on%20Adaptive%20Grid%20for%20Reinforcement%20Learning/plots/Maze_hard_good.png) |

📁 [View this project](./Breadth-First%20Exploration%20on%20Adaptive%20Grid%20for%20Reinforcement%20Learning/README.md)

---

## 🤖 Machine Learning Projects (`/ML`)

### 📂 `SVM`, `Perceptron`, etc.

Implementation of classical ML models from scratch.  
Emphasis on understanding optimization logic, loss functions, and convergence behavior.

---

### 📂 `Android Malware Detection`

A Kaggle classification project aiming to detect malicious Android applications.

#### 🔍 Highlights:
- Explored multiple models:  
  - Decision Tree (and variants)  
  - Perceptron (and variants)  
  - Support Vector Machine (SVM)  
  - Ensemble methods: Adaboost + SVM/DT/Perceptron
- Final model: **Adaboost with SVM + DT + Perceptron**  
- Achieved **~90% accuracy on Kaggle**

📁 [View this project](./ML/Android-Malware-Detection/README.md)

---

## 🧰 Tech Stack

- Python, NumPy, Matplotlib, Scikit-learn
- Gym environments (for RL extensions)
- Kaggle (for data competition)

---

