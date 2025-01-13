import numpy as np


class Maze:
    def __init__(self, size):
        self.states = [(x, y) for x in range(size) for y in range(size)]
        self.size = size
        self.goal_state = (size - 1, size - 1)
        self.actions = ['Up', 'Down', 'Left', 'Right']

    def is_goal_state(self, state):
        return state == self.goal_state

    def step(self, state, action):
        x, y = state
        if action == 'Up':
            x = max(x - 1, 0)
        elif action == 'Down':
            x = min(x + 1, self.size - 1)
        elif action == 'Left':
            y = max(y - 1, 0)
        elif action == 'Right':
            y = min(y + 1, self.size - 1)
        next_state = (x, y)
        reward = 1 if self.is_goal_state(next_state) else 0
        return next_state, reward


class DPAgent:
    def __init__(self, maze, alpha, gamma):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {(state, action): 0.0 for state in self.maze.states for action in self.maze.actions}

    def dp_q_learning(self, max_iters=100):
        for iteration in range(max_iters):
            print(f"Iteration {iteration + 1}")
            delta = 0
            for state in self.maze.states:
                for action in self.maze.actions:
                    next_state, reward = self.maze.step(state, action)
                    old_value = self.q_values[(state, action)]
                    future_values = [self.q_values[(next_state, a)] for a in self.maze.actions]
                    best_future_value = max(future_values)
                    new_value = reward + self.gamma * best_future_value
                    self.q_values[(state, action)] = (1 - self.alpha) * old_value + self.alpha * new_value
                    delta = max(delta, abs(new_value - old_value))
                    print(f"Updated Q-value for state {state} and action {action}: {new_value:.2f}")
            if delta < 1e-6:  # Convergence criterion
                print("Q-values converged.")
                break
        print("Final Q-values after DP Q-learning:")
        for state_action, value in self.q_values.items():
            print(f"Q{state_action}: {value:.2f}")


# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
maze_size = 3  # Size of the maze (3x3)

# Initialize maze and agent
maze = Maze(maze_size)
dp_agent = DPAgent(maze, alpha, gamma)

# Run DP Q-learning
dp_agent.dp_q_learning(max_iters=3)  # Set to 5 for initial debugging, increase as needed
