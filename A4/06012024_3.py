"""
epsilon Qlearning implementation
"""

import numpy as np

class MiniMaze:
    """
    Represents a simple grid world maze where an agent moves to reach a terminal goal state.
    """
    def __init__(self, size=3):
        self.size = size  # Defines the size of the maze (size x size)
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
    """
    Agent that learns to navigate the MiniMaze using Q-Learning algorithms.
    It can use either Dynamic Programming (DP) or Reinforcement Learning (RL) strategies.
    """
    def __init__(self, maze, discount_factor=0.9, max_iters=100, alpha=0.2, epsilon=0.9, epsilon_decay=1,
                 epsilon_min=0.01, qvalues_print_frequency=50):
        self.maze = maze  # The MiniMaze environment
        self.gamma = discount_factor  # Discount factor for future rewards

        # Parameters for Dynamic Programming (DP)
        self.max_iters = max_iters  # Maximum iterations for DP learning

        # Parameters for Reinforcement Learning (RL)
        self.alpha = alpha  # Learning rate for RL
        self.epsilon = epsilon  # Exploration rate for RL
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon per episode for RL
        self.epsilon_min = epsilon_min  # Minimum exploration rate for RL
        self.print_frequency = qvalues_print_frequency

        # Initialize Q-values dictionary for all state-action pairs
        self.Q_values = {(state, action): 0 for state in maze._get_all_states() for action in ['up', 'down', 'left', 'right']}

    def reset_q_values(self):
        """
        Resets the Q-values to the initial state for a fresh start.
        """
        self.Q_values = {(state, action): 0 for state in self.maze._get_all_states() for action in
                         ['up', 'down', 'left', 'right']}

    ##############################################
    # Methods utilized by the DP Qlearning agent #
    ##############################################
    def DP_tabular(self):
        """
        Learns an optimal policy using Dynamic Programming (DP) tabular method.
        It iteratively applies the Bellman equation to find the optimal Q-values.
        """

        self.reset_q_values()  # Reset Q-values before starting DP

        iter_num = 0

        print()
        print("=========================")
        print("= DP Qlearning training =")
        print("=========================")

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
        print("\n\tFinal Q-Values:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                q_values = [self.Q_values[(state, action)] for action in ['up', 'down', 'left', 'right']]
                # Print Q-values in a custom format
                print(
                    f"({x}, {y}): up={q_values[0]:.2f}, down={q_values[1]:.2f}, left={q_values[2]:.2f}, right={q_values[3]:.2f} | ",
                    end="")
            print()  # Newline after each row
        print("======================================================================================================"
              "=====================================================")

    ###################################################
    # Methods utilized by the epsilon Qlearning agent #
    ###################################################
    def rl_epsilon_q_learning(self, episodes):
        self.reset_q_values()  # Reset Q-values before starting RL
        last_Q_values = self.Q_values.copy()  # To store Q-values from the previous episode
        converged_episode = None  # To record the episode number when convergence occurs

        print()
        print("==============================")
        print("= epsilon Qlearning training =")
        print("==============================")

        for episode in range(episodes):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Decrease epsilon
            self.maze.state = self.maze.reset()  # Reset the maze to the start state
            done = False
            while not done:
                action = self.select_action(self.maze.state)
                new_state, reward = self.maze.step(action)
                # Update Q-values immediately after taking an action
                self.update_q_table(self.maze.state, action, reward, new_state)
                done = new_state is None  # Check if the episode has ended
                if not done:
                    self.maze.state = new_state  # Move to the new state

            # Check for convergence
            max_diff = max(abs(self.Q_values[key] - last_Q_values[key]) for key in self.Q_values)
            last_Q_values = self.Q_values.copy()
            if max_diff < 1e-6 and not converged_episode:
                converged_episode = episode + 1

            if episode < 5 or episode % self.print_frequency == 0 or episode == episodes - 1:
                self.visualize_rl_training(episode + 1)

        self.visualize_final_policy_and_q_values()

        # Print information about convergence if it occurred
        if converged_episode:
            print(f"\nConvergence occurred at episode {converged_episode}.")
            print("\nConverged Q-Values:")
            for y in reversed(range(self.maze.size)):
                for x in range(self.maze.size):
                    state = (x, y)
                    q_values = [f"{self.Q_values[(state, action)]:.3f}" for action in ['up', 'down', 'left', 'right']]
                    print(f"({x}, {y}): {q_values} | ", end="")
                print()  # Newline after each row
            print(
                "======================================================================================================"
                "======================================")

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.choice(['up', 'down', 'left', 'right'])
        else:
            # Exploitation: best action based on current Q-values
            state_actions = [(state, action) for action in ['up', 'down', 'left', 'right']]
            return max(state_actions, key=lambda x: self.Q_values[x])[1]

    def update_q_table(self, state, action, reward, new_state):
        if new_state:
            # If not in the terminal state, consider future rewards
            future_rewards = [self.Q_values[(new_state, next_action)] for next_action in ['up', 'down', 'left', 'right']]
            best_future_value = max(future_rewards)
        else:
            # In the terminal state, there are no future rewards
            best_future_value = 0

        # Bellman equation update
        self.Q_values[(state, action)] *= (1 - self.alpha)
        self.Q_values[(state, action)] += self.alpha * (reward + self.gamma * best_future_value)

    def visualize_rl_training(self, episode):
        print(f"\nQ-values after episode {episode}:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                q_values = [self.Q_values[(state, action)] for action in ['up', 'down', 'left', 'right']]
                print(f"({x}, {y}): {q_values} | ", end="")
            print()  # Newline after each row

    def visualize_final_policy_and_q_values(self):
        # Value function visualization
        print("\nFinal Value Function:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                best_value = max(self.Q_values[(state, a)] for a in ['up', 'down', 'left', 'right'])
                print(f"{best_value:.2f} ", end="")
            print()

        # Final policy visualization
        print("\nFinal Policy:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                if state == self.maze.terminal_state:
                    print("*    ", end="")
                else:
                    actions = ['up', 'down', 'left', 'right']
                    best_action_value = max(self.Q_values[(state, a)] for a in actions)
                    # Considering actions optimal if within a small margin of the best action value
                    margin = 0.01
                    best_actions = [a for a in actions if self.Q_values[(state, a)] >= best_action_value - margin]
                    arrows = ''.join(self.action_to_arrow(a) for a in best_actions).ljust(4, ' ')
                    print(f"{arrows} ", end="")
            print()

    def visualize_rl_training(self, episode):
        print(f"\nQ-values after episode {episode}:")
        for y in reversed(range(self.maze.size)):
            for x in range(self.maze.size):
                state = (x, y)
                q_values = [f"{self.Q_values[(state, action)]:.3f}" for action in ['up', 'down', 'left', 'right']]
                print(f"({x}, {y}): {q_values} | ", end="")
            print()  # Newline after each row

# Initialize the MiniMaze and the QLearningAgent
maze = MiniMaze(size=3)
agent = QLearningAgent(maze,epsilon= 0.9,alpha=0.9)

# Perform the learning process
agent.DP_tabular()


# Perform RL Epsilon Q-learning
agent.rl_epsilon_q_learning(episodes=500)

print("\n\ndone")