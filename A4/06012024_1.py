"""
DP finished
"""

import numpy as np

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
    def __init__(self, maze, discount_factor=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, max_iters=1000):
        self.maze = maze
        self.gamma = discount_factor
        self.alpha = alpha
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q_values = {(state, action): 0 for state in maze._get_all_states() for action in ['up', 'down', 'left', 'right']}
        self.max_iters = max_iters
        self.epsilon = epsilon

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

             # Print the value function and policy
            print(f"\ni={iter_num}")
            self.print_policy_and_value()

            if max_diff < 1e-6:
                print(f"\n\n\nDP tabular Q-learning case:\n\tQ-values converged at iteration {iter_num}.")
                break

            iter_num += 1

        # Once done, print the final Q-values in a custom format
        self.print_final_q_values()

    def print_policy_and_value(self):
        print("Value Function:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                best_value = max(self.Q_values[(state, a)] for a in ['up', 'down', 'left', 'right'])
                print(f"{best_value:.2f} ", end="")
            print()

        print("Resulting Policy:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                if state == self.maze.terminal_state:
                    print("*    ", end="")
                else:
                    actions = ['up', 'down', 'left', 'right']
                    best_action_value = max(self.Q_values[(state, a)] for a in actions)
                    best_actions = [a for a in actions if self.Q_values[(state, a)] == best_action_value]
                    # Convert actions to arrows
                    arrows = ''.join(self.action_to_arrow(a) for a in best_actions).ljust(4, ' ')
                    print(f"{arrows} ", end="")
            print()

    def action_to_arrow(self, action):
        return {
            'up': '^',
            'down': 'v',
            'left': '<',
            'right': '>'
        }.get(action, '?')

    def print_final_q_values(self):
        print("\nFinal Q-Values:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                q_values = [self.Q_values[(state, action)] for action in ['up', 'down', 'left', 'right']]
                # Print Q-values in a custom format
                print(
                    f"({x}, {y}): up={q_values[0]:.2f}, down={q_values[1]:.2f}, left={q_values[2]:.2f}, right={q_values[3]:.2f} | ",
                    end="")
            print()  # Newline after each row

    def rl_epsilon_q_learning(self, episodes):
        for episode in range(episodes):
            print(episode)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            state = self.maze.reset()
            done = False
            while not done:
                action = self.select_action(state)
                new_state, reward = self.maze.step(action)
                done = new_state is None
                self.update_q_table(state, action, reward, new_state)
                state = new_state if new_state is not None else state

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(['up', 'down', 'left', 'right'])
        else:
            state_actions = [(state, action) for action in ['up', 'down', 'left', 'right']]
            return max(state_actions, key=lambda x: self.Q_values[x])[1]

    def update_q_table(self, state, action, reward, new_state):
        best_future_value = 0 if new_state is None else max(
            self.Q_values[(new_state, a)] for a in ['up', 'down', 'left', 'right'])
        self.Q_values[(state, action)] = (1 - self.alpha) * self.Q_values[(state, action)] + self.alpha * (
                    reward + self.gamma * best_future_value)


# Initialize the MiniMaze and the QLearningAgent
maze = MiniMaze(size=3)
agent = QLearningAgent(maze)

# Perform the learning process
agent.DP_tabular()

print("\n\ndone")

# Perform RL Epsilon Q-learning
agent.rl_epsilon_q_learning(episodes=100)