"""
Ángel Amando Gil Álamo - 2788988
"""
# Dynamic Programming solution for the stochastic knapsack problem.
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the knapsack problem
T = 10  # Number of items
W = 10  # Total weight limit of the knapsack
num_simulations = 1000  # Number of simulations

# Initialize the value function table
V = np.zeros((T+1, W+1))  # +1 because we include 0 state and 0 weight
policy = np.zeros((T, W+1))

# Populate the value function using DP with the insight form a)
for t in range(T-1, -1, -1):  # Start from last item, go to first item (backward)
    for w in range(W, -1, -1):  # For each possible remaining capacity
        # Compute the largest item size that is beneficial to take
        max_fit = w
        while max_fit > 0 and V[t+1][w-max_fit] + 1 < V[t+1][w]:
            max_fit -= 1
        policy[t][w] = max_fit
        
        # Compute the expected value for each possible action
        expected_value_take = 0
        expected_value_not_take = V[t + 1][w]
        for size in range(1, W + 1):
            prob = 0.1  # Probability for each size
            if size <= max_fit:
                expected_value_take += prob * (1 + V[t + 1][w - size])
            else:
                expected_value_take += prob * V[t + 1][w]

        # Update the value function
        V[t][w] = max(expected_value_not_take, expected_value_take)


# Output the final value function table
print("Value Function Table:")
print(V)

# Calculate the maximal expected reward
max_expected_reward = V[0][W]
print("Maximal Expected Reward:", max_expected_reward)


# Policy heatmap
plt.figure(figsize=(10, 8))
plt.imshow(policy/10, cmap='Blues', aspect='auto')
plt.colorbar(label='Take Item (1) or Not (0)')
plt.title('Optimal Policy Heatmap')
plt.xlabel('Weight of knapsack')
plt.ylabel('Item Number')
plt.xticks(range(W+1), labels=[str(10-i) for i in range(0, W+1)])
plt.yticks(range(T), labels=[str(11-i) for i in range(1, T+1)])

plt.show()

# Simulate the process under the optimal policy
rewards = np.zeros(num_simulations)
for simulation in range(num_simulations):
    remaining_capacity = W
    total_reward = 0
    for t in range(T):
        item_size = np.random.randint(1, W+1)  # Random item size
        if remaining_capacity >= item_size and policy[t][remaining_capacity] >= item_size:
            # If taking the item is within policy, update reward and capacity
            total_reward += 1
            remaining_capacity -= item_size
    rewards[simulation] = total_reward

print("Average Simulation Reward:",np.average(rewards))
print("Std of the Simulation Reward:",np.std(rewards))

# Histogram of the simulation rewards
plt.figure(figsize=(10, 6))
plt.hist(rewards, bins=range(int(min(rewards)), int(max(rewards)) + 2), align='left', color='blue', edgecolor='black')
plt.title('Histogram of Rewards from Simulations')
plt.xlabel('Total Reward')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.xticks(range(int(min(rewards)), int(max(rewards)) + 1))
plt.show()