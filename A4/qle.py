# Attempting to correct the code and run it again based on the user's specifications.

import numpy as np

class MiniMaze:
    def __init__(self, size=3):
        self.size = size
        self.terminal_state = (self.size - 1, self.size - 1)  # Terminal state at top-right corner
        self.state = (0, 0)  # Start at bottom-left corner

    def __str__(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.state:             # Agent's current location
                    print('S ')
                elif (i, j) == self.terminal_state:  # Goal state
                    print('R ')
                else:                                # Empty cell
                    print('. ')
            print()
        print()

    def step(self, action):
        # If state is terminal, end the episode
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

        self.state = (x, y)

        return self.state, 0

    def reset(self):
        self.state = (0, 0)
        return self.state

class QLearningAgent:
    def __init__(self, maze, discount_factor=0.9, learning_rate=0.1):
        self.maze = maze
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.Q_values = {(state, action): 0 for state in self._get_all_states() for action in ['up', 'down', 'left', 'right']}

    def _get_all_states(self):
        return [(x, y) for x in range(self.maze.size) for y in range(self.maze.size)]

    def learn(self):
        for state in self._get_all_states():
            for action in ['up', 'down', 'left', 'right']:
                self.maze.state = state
                new_state, reward = self.maze.step(action)
                self.update_q_table(state, action, new_state, reward)

    def update_q_table(self, state, action, new_state, reward):
        if new_state is not None:
            reward += self.gamma * np.max(self.Q_values[(state, action)])

        self.Q_values[(state, action)] *= (1 - self.alpha)
        self.Q_values[(state, action)] += self.alpha * reward

    def get_policy(self):
        policy = {state: max(['up', 'down', 'left', 'right'], key=lambda action: self.Q_values[(state, action)])
                  for state in self._get_all_states() if not state == self.maze.terminal_state}
        return policy

# Initialize the MiniMaze and the QLearningAgent
maze = MiniMaze(size=3)
agent = QLearningAgent(maze)

# Perform the learning process
for i in range(5):
    agent.learn()

# Retrieve the policy from the learned Q-values
policy = agent.get_policy()

policy  # Display the learned policy

print(agent.Q_values)
print(policy)
