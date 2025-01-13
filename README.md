# Dynamic Programming and Reinforcement Learning Projects Overview

This repository contains assignments completed for the **Dynamic Programming and Reinforcement Learning** course. The projects apply dynamic programming and reinforcement learning techniques to solve optimization problems in stochastic and game-based environments.

---

## Assignments

### **Assignment 1: Stochastic Knapsack Problem**
- **Objective**: Solve a knapsack problem where items of random sizes arrive sequentially.
- **Key Highlights**:
  - Formulated the problem as a dynamic programming task.
  - Derived the optimal policy and implemented simulations to validate the approach.
  - Visualized results using policy heatmaps and reward histograms.

### **Assignment 2: System Replacement with Preventive Policies**
- **Objective**: Model a deteriorating system using a Markov Reward Chain (MRC) and optimize replacement policies.
- **Key Highlights**:
  - Computed stationary distributions and long-run average costs.
  - Solved the problem using policy iteration and value iteration.
  - Evaluated preventive replacement strategies to minimize costs.

### **Assignment 3: Connect 4 with Monte Carlo Tree Search**
- **Objective**: Implement a Monte Carlo Tree Search (MCTS) algorithm to defeat a random agent in the game of Connect 4.
- **Key Highlights**:
  - Encoded the game state and implemented the MCTS with UCB for action selection.
  - Analyzed convergence and visualized winning probabilities during gameplay.

### **Assignment 4: Mini-Maze Q-Learning**
- **Objective**: Solve a mini-maze environment using Q-learning and explore function approximators.
- **Key Highlights**:
  - Tabular Q-learning with ε-greedy policy for a 3x3 maze.
  - Extended to a 5x5 maze with advanced configurations and hyperparameter tuning.
  - Optional implementation of function approximators for the highest grade.

---

## Repository Structure
- **`Assignment1/`**: Stochastic knapsack problem implementation.
- **`Assignment2/`**: MRC and optimal replacement policies.
- **`Assignment3/`**: Connect 4 implementation using MCTS.
- **`Assignment4/`**: Mini-maze problem solved with Q-learning.

---

## Tools and Technologies
- **Python**: Core programming language for all implementations.
- **Libraries**:
  - `NumPy`: Numerical computations.
  - `matplotlib`: Visualizations.
  - `Gymnasium`: For environment simulations (where applicable).
