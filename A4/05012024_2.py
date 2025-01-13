"""
THIS VERSION IS FIRST OF DP TABULAR QLEARNING WORKING AS IT SHOULD!!
"""


class MiniMaze:
    def __init__(self, size=3):
        self.size = size
        self.terminal_state = (self.size - 1, self.size - 1)  # Terminal state at top-right corner
        self.state = (0, 0)  # Start at bottom-left corner

    def step(self, action):
        # If state is terminal, end the episode and return the reward
        if self.state == self.terminal_state:
            return None, 1

        # Otherwise, update the agent's position preventing going off-grid
        x, y = self.state
        if action == 'up':
            y = min(self.size - 1, y + 1)
        elif action == 'down':
            y = max(0, y - 1)
        elif action == 'left':
            x = max(0, x - 1)
        elif action == 'right':
            x = min(self.size - 1, x + 1)

        new_state = (x, y)

        return new_state, 0

    def reset(self):
        self.state = (0, 0)
        return self.state

    def _get_all_states(self):
        return [(x, y) for x in range(self.size) for y in range(self.size)]


class QLearningAgent:
    def __init__(self, maze, discount_factor=0.9, max_iters=50):
        self.maze = maze
        self.gamma = discount_factor
        self.Q_values = {(state, action): 0 for state in self.maze._get_all_states() for action in ['up', 'down', 'left', 'right']}
        self.max_iters = max_iters

    def learn(self):
        for state in self.maze._get_all_states():
            for action in ['up', 'down', 'left', 'right']:
                if state != (2, 2):
                    new_state, reward = self.maze.step(action)
                    # Bellman update for deterministic policy
                    best_future_value = max(self.Q_values[(new_state, a)] for a in ['up', 'down', 'left', 'right'])
                    self.Q_values[(state, action)] = reward + self.gamma * best_future_value

    def DP_tabular(self):
        iter_num = 0
        while iter_num < self.max_iters:
            new_Q_values = self.Q_values.copy()
            for state in self.maze._get_all_states():
                for action in ['up', 'down', 'left', 'right']:
                    self.maze.state = state
                    new_state, reward = self.maze.step(action)
                    if new_state != None:
                        best_future_value = max(self.Q_values[(new_state, a)] for a in ['up', 'down', 'left', 'right'])
                    else:
                        best_future_value = 0
                    new_Q_values[(state, action)] = reward + self.gamma * best_future_value

            # Check for convergence by comparing old and new Q-values
            max_diff = max(abs(new_Q_values[key] - self.Q_values[key]) for key in self.Q_values)
            self.Q_values = new_Q_values

            # Print the Q-values for this iteration
            print(f"Iteration {iter_num + 1}:")
            for state_action, value in self.Q_values.items():
                print(f"  Q{state_action}: {value:.2f}")

            if max_diff < 1e-6:
                print(f"DP tabular Q-values converged at iter {iter_num}.")
                break

            iter_num += 1

    def get_policy(self):
        policy = {state: max(['up', 'down', 'left', 'right'], key=lambda action: self.Q_values[(state, action)])
                  for state in self.maze._get_all_states() if not state == self.maze.terminal_state}
        return policy


# Initialize the MiniMaze and the QLearningAgent
maze = MiniMaze(size=3)
agent = QLearningAgent(maze)

# Perform the learning process
agent.DP_tabular()

# Retrieve the policy from the learned Q-values
policy = agent.get_policy()

# Print the Q-values and the policy
print("Learned Q-values:")
for key, value in sorted(agent.Q_values.items()):
    print(f"{key}: {value:.2f}")

print("\nLearned policy:")
for state, action in sorted(policy.items()):
    print(f"State {state}: {action}")