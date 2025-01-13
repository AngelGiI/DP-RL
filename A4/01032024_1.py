# Revised code for Part 1 with both DP tabular Q-learning and RL epsilon Q-learning agents.
# Additionally, it includes a visualization of the Q-table and an early stopping mechanism for convergence.

import numpy as np
import matplotlib.pyplot as plt

# Constants
SIZE = 3  # Change to 5 for part 2
ALPHA = 0.25
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
ACTIONS = ['Up', 'Down', 'Left', 'Right']
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Deltas for Up, Down, Left, Right

# Maze Environment
class Maze:
    def __init__(self, size):
        self.size = size
        self.goal_state = (size - 1, size - 1)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        if self.state == self.goal_state:
            return 1, self.goal_state  # Reward, Terminal state

        # Calculate new state
        next_state = (max(min(self.state[0] + ACTION_DELTAS[action][0], self.size - 1), 0),
                      max(min(self.state[1] + ACTION_DELTAS[action][1], self.size - 1), 0))

        # Check if the goal state is reached
        reward = 1 if next_state == self.goal_state else 0
        self.state = next_state
        return reward, next_state

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_values = np.zeros((env.size, env.size, len(ACTIONS)))
        self.epsilon = EPSILON

    # Dynamic Programming Q-Learning
    def dp_q_learning(self, alpha, gamma, threshold):
        while True:
            delta = 0
            for state_x in range(self.env.size):
                for state_y in range(self.env.size):
                    for action, action_delta in enumerate(ACTION_DELTAS):
                        next_state = (max(min(state_x + action_delta[0], self.env.size - 1), 0),
                                      max(min(state_y + action_delta[1], self.env.size - 1), 0))
                        reward = 1 if next_state == self.env.goal_state else 0
                        old_value = self.q_values[state_x, state_y, action]
                        next_max = np.max(self.q_values[next_state[0], next_state[1]])
                        self.q_values[state_x, state_y, action] = old_value + alpha * (reward + gamma * next_max - old_value)
                        delta = max(delta, np.abs(old_value - self.q_values[state_x, state_y, action]))
            if delta < threshold:
                break

    # Reinforcement Learning Epsilon Q-Learning
    def rl_q_learning(self, alpha, gamma, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(range(4))
                else:
                    action = np.argmax(self.q_values[state[0], state[1]])
                reward, next_state = self.env.step(action)
                done = next_state == self.env.goal_state
                old_value = self.q_values[state[0], state[1], action]
                next_max = np.max(self.q_values[next_state[0], next_state[1]])
                self.q_values[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)
                state = next_state
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # Print Q-values for all states and actions
    def print_q_values(self):
        print("Maze Q-Values:")
        for y in range(self.env.size):
            for x in range(self.env.size):
                print(f"State ({x}, {y}):", end=" ")
                for action in range(len(ACTIONS)):
                    print(f"{ACTIONS[action]}={self.q_values[x, y, action]:.2f}", end=" ")
                print()
        print()

    # Visualize the Q-table with arrows for the optimal policy
    def visualize_optimal_policy(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.env.size - 0.5)
        ax.set_ylim(-0.5, self.env.size - 0.5)
        ax.grid(which='both')
        ax.set_xticks(np.arange(0, self.env.size))
        ax.set_yticks(np.arange(0, self.env.size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Adding arrows to indicate the optimal policy at each state
        for x in range(self.env.size):
            for y in range(self.env.size):
                best_action = np.argmax(self.q_values[x, y])
                if best_action == 0:  # Up
                    ax.arrow(x, y, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_action == 1:  # Down
                    ax.arrow(x, y, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_action == 2:  # Left
                    ax.arrow(x, y, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_action == 3:  # Right
                    ax.arrow(x, y, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

        plt.title('Optimal Policy Visualization')
        plt.show()

    def check_convergence(self, threshold=1e-3):
        is_converged = True
        for x in range(self.env.size):
            for y in range(self.env.size):
                best_action = np.argmax(self.q_values[x, y])
                if not np.isclose(self.q_values[x, y, best_action], 1, atol=threshold):
                    is_converged = False
        return is_converged

# Initialize the maze and agent
maze_env = Maze(SIZE)
agent = QLearningAgent(maze_env)

# Run DP Q-Learning and check convergence
print("DP Q-Learning")
agent.dp_q_learning(ALPHA, GAMMA, threshold=0.01)
agent.print_q_values()
print("Convergence:", agent.check_convergence())

# Run RL Epsilon Q-Learning
print("RL Q-Learning")
agent.rl_q_learning(ALPHA, GAMMA, episodes=1000)
agent.print_q_values()

# Visualize the policy on the grid
agent.visualize_optimal_policy()