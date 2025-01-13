##########################################################################################################################
####                                                                                                                  ####
####                                                        PART 1                                                    ####
####                                                                                                                  ####
##########################################################################################################################

import numpy as np
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


################################################
# 1. Initialize MC
################################################

def init_Q():
    return np.zeros((5, 2))


def init_chain():
    # for each state, action 1 and action 2
    transitions = {0: [1, 0],
                   1: [2, 0],
                   2: [3, 0],
                   3: [4, 0],
                   4: [4, 0]}

    # rewards for each state's actions
    R = np.array([[0, 0.2],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0]])

    return transitions, R


################################################
# 2. Tabular Q-learning (dynamic programming)
################################################

def BK(state, old_Q, action, next_state, R, gamma):
    # tabular values of
    A = old_Q[next_state, :]

    # immediate reward of the current state, for either action 1 or action 2
    R_xa = R[state][action]

    return R_xa + gamma * max(A)


def tabular_Qlearning(Q, transitions, R, gamma):
    # loop until converged
    converged = False
    while not converged:

        # save copy of the Q in the previous timestep
        old_Q = deepcopy(Q)

        # iterate over all states
        for state in range(len(Q)):
            # retrieve all possible next states
            next_states = np.array(transitions[state])

            # calculate BK(state, action) for action 1 and action 2
            a1, a2 = [BK(state, old_Q, action, next_state, R, gamma)
                      for (action, next_state) in enumerate(next_states)]

            # update Q with BK(state, action )
            Q[state] = a1, a2

        # stop if the Q has converged
        if np.array_equal(Q, old_Q):
            converged = True

    return Q


################################################
# 3. Q-learning with 5 states (epsilon greedy)
################################################

# initialize Q and the chain
Q = init_Q()
transitions, R = init_chain()

# initialize hyperparameters
gamma = .9

Q = tabular_Qlearning(Q, transitions, R, gamma)
print(Q)


def init_Q():
    return np.zeros((5, 2))


def init_chain():
    # for each state, action 1 and action 2
    transitions = {0: [1, 0],
                   1: [2, 0],
                   2: [3, 0],
                   3: [4, 0],
                   4: [4, 0]}

    # rewards for each state's actions
    R = np.array([[0, 0.2],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0]])

    return transitions, R


def update_Q(Q, state, action, next_state, R, gamma, alpha, epsilon):
    # current Q(state, action) value
    Q_xa = Q[state][action]

    # Q(next_state, action) value
    Q_ya = Q[next_state]

    # immediate reward of the current state, for either action 1 or action 2
    R_xa = R[state][action]

    return Q_xa + alpha * (R_xa + gamma * max(Q_ya) - Q_xa), R_xa


def Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps):
    episode_history = []

    for episode in range(episodes):

        # reset starting state
        state = 0

        step_history = []

        for step in range(steps):

            # retrieve all possible next states
            next_states = transitions[state]

            # greedy action
            if np.random.sample() < (1 - epsilon):
                action = np.argmax(Q[state])
            # random action
            else:
                action = np.random.choice([0, 1])

            # take action
            next_state = next_states[action]

            # update Q(state, action) with the new Q-value and remember R(state, action)
            Q[state][action], R_xa = update_Q(Q, state, action, next_state, R, gamma, alpha, epsilon)

            # save R(state, action) history
            step_history.append(R_xa)

            # update state
            state = next_state

        # save mean (R(state, action)) per episode
        episode_history.append(np.mean(step_history))

    return Q, np.array(episode_history)


# initialize Q and the chain
Q = init_Q()
transitions, R = init_chain()

# initialize hyperparameters
gamma = .9
alpha = 0.9
epsilon = 0.1

episodes, steps = 500, 50
Q, episode_history = Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps)
Q

# initialize hyperparameters
gamma = .9
epsilon = 0.1

episodes, steps = 100, 20

results = []
df = pd.DataFrame()

for alpha in [0.1, 0.3, 0.9]:

    print(alpha)
    avg_rewards = []

    # repeat for some number of iterations
    for _ in range(100):
        # initialize Q and the chain
        Q = init_Q()
        transitions, R = init_chain()

        # calculate the average rewards
        avg_rewards = Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps)[1]

        # add them to a dataframe
        df1 = pd.DataFrame()
        df1['Episode'] = np.arange(episodes) + 1
        df1['Average reward'] = avg_rewards
        df1['alpha'] = alpha
        df1['epsilon'] = epsilon

        # concatenate this iteration's average rewards to the other average rewards
        df = pd.concat([df, df1], ignore_index=True)

# make a lineplot with averages and CI
sns.lineplot(data=df, x="Episode", y="Average reward", hue="alpha", ci=95)

# plt.savefig(f'alpha_5_states_{episodes,steps,epsilon}.png', bbox_inches='tight', dpi=300)
plt.show()

# initialize hyperparameters
gamma = .9
alpha = .9

episodes, steps = 100, 50

results = []
df = pd.DataFrame()

for epsilon in [0.01, 0.1, 0.2]:

    print(epsilon)
    avg_rewards = []

    # repeat for some number of iterations
    for _ in range(100):
        # initialize Q and the chain
        Q = init_Q()
        transitions, R = init_chain()

        # calculate the average rewards
        avg_rewards = Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps)[1]

        # add them to a dataframe
        df1 = pd.DataFrame()
        df1['Episode'] = np.arange(episodes) + 1
        df1['Average reward'] = avg_rewards
        df1['alpha'] = alpha
        df1['epsilon'] = epsilon

        # concatenate this iteration's average rewards to the other average rewards
        df = pd.concat([df, df1], ignore_index=True)

# make a lineplot with averages and CI
sns.lineplot(data=df, x="Episode", y="Average reward", hue="epsilon", ci=95)

# plt.savefig(f'epsilon_5_states_{episodes,steps,alpha}.png', bbox_inches='tight', dpi=300)
plt.show()


################################################
# 4. Q-learning with 10 states (epsilon greedy)
################################################

def init_Q_10():
    return np.zeros((10, 2))


def init_chain_10():
    # for each state, action 1 and action 2
    transitions = {0: [1, 0],
                   1: [2, 0],
                   2: [3, 0],
                   3: [4, 0],
                   4: [5, 0],
                   5: [6, 0],
                   6: [7, 0],
                   7: [8, 0],
                   8: [9, 0],
                   9: [9, 0]}

    # rewards for each state's actions
    R = np.array([[0, 0.2],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0]])

    return transitions, R


# initialize Q and the chain
Q = init_Q_10()
transitions, R = init_chain_10()

# initialize hyperparameters
gamma = .9
alpha = 0.9
epsilon = 0.1

episodes, steps = 500, 500

Q, episode_history = Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps)
print(Q)

# initialize hyperparameters
gamma = .9
epsilon = 0.1

episodes, steps = 100, 50

results = []
df = pd.DataFrame()

for alpha in [0.1, 0.3, 0.9]:

    print(alpha)
    avg_rewards = []

    # repeat for some number of iterations
    for _ in range(100):
        # initialize Q and the chain
        Q = init_Q_10()
        transitions, R = init_chain_10()

        # calculate the average rewards
        avg_rewards = Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps)[1]

        # add them to a dataframe
        df1 = pd.DataFrame()
        df1['Episode'] = np.arange(episodes) + 1
        df1['Average reward'] = avg_rewards
        df1['alpha'] = alpha
        df1['epsilon'] = epsilon

        # concatenate this iteration's average rewards to the other average rewards
        df = pd.concat([df, df1], ignore_index=True)

# make a lineplot with averages and CI
sns.lineplot(data=df, x="Episode", y="Average reward", hue="alpha", ci=95)

# plt.savefig(f'alpha_10_states_{episodes,steps,epsilon}.png', bbox_inches='tight', dpi=300)
plt.show()

# initialize hyperparameters
gamma = .9
alpha = .9

episodes, steps = 100, 50

results = []
df = pd.DataFrame()

for epsilon in [0.01, 0.1, 0.2]:

    print(epsilon)
    avg_rewards = []

    # repeat for some number of iterations
    for _ in range(100):
        # initialize Q and the chain
        Q = init_Q_10()
        transitions, R = init_chain_10()

        # calculate the average rewards
        avg_rewards = Qlearning(Q, transitions, R, gamma, alpha, epsilon, episodes, steps)[1]

        # add them to a dataframe
        df1 = pd.DataFrame()
        df1['Episode'] = np.arange(episodes) + 1
        df1['Average reward'] = avg_rewards
        df1['alpha'] = alpha
        df1['epsilon'] = epsilon

        # concatenate this iteration's average rewards to the other average rewards
        df = pd.concat([df, df1], ignore_index=True)

# make a lineplot with averages and CI
sns.lineplot(data=df, x="Episode", y="Average reward", hue="epsilon", ci=95)

# plt.savefig(f'epsilon_10_states_{episodes,steps,alpha}.png', bbox_inches='tight', dpi=300)
plt.show()

##########################################################################################################################
####                                                                                                                  ####
####                                                        PART 2                                                    ####
####                                                                                                                  ####
##########################################################################################################################


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Configuring Pytorch
from collections import namedtuple, deque
from itertools import count
import random
import numpy as np
import pandas as pd

ALPHA = 0.1
GAMMA = 0.9
EPSILION = 0.9
n_state = 10
ACTIONS = ['left', 'right']
num_train_episode = 1200
obs_dim = 1
FRESH_TIME = 0.1
num_steps = 50


class chain_env:
    def __init__(self, n_state):
        self.num_step = 0
        self.state = 0
        self.n_state = n_state
        self.actions = ACTIONS

    def update_env(self, step_counter):
        env = ['-'] * (self.n_state - 1) + ['T']
        if self.state == 'terminal':
            final_env = ['-'] * (self.n_state - 1) + ['T']
            return True, step_counter
        else:
            env[int(self.state)] = '*'
            env = ''.join(env)
            # print(env)
            return False, step_counter

    def step(self, acindex):
        action = self.actions[acindex]
        if action == 'right':
            if self.state == self.n_state - 1:
                next_state = self.state
                reward = 1.0
            else:
                next_state = self.state + 1
                reward = 0.0
        else:
            if self.state == 0:
                next_state = 0.0
                reward = 0.2
            else:
                next_state = 0.0
                reward = 0.0
        self.state = next_state
        self.update_env(self.num_step)
        self.num_step += 1
        return np.array(next_state).reshape(-1), reward, False, {}

    def reset(self):
        self.state = 0
        return np.array(self.state).reshape(-1)

    def num_actions(self):
        return 2

    def num_states(self):
        return 1


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


setup_seed(1256)

'''
'''
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(inputs, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden - 1)])
        self.output_layer = nn.Linear(hidden_size, outputs)

    def forward(self, x):
        x.to(device)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class Agent():
    def __init__(self, env, state_dim, n_actions):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.num_hidden_layers = 1
        self.size_hidden_layers = 8
        self.target_update_freq = 100

        self.iter_steps = 0
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.env = env
        self.epsilon = 1
        self.eps_min = 0.1
        self.eps_dec = 5e-5

        self.policy_net = DQN(self.state_dim, self.n_actions, self.num_hidden_layers, self.size_hidden_layers).to(
            device)
        self.target_net = DQN(self.state_dim, self.n_actions, self.num_hidden_layers, self.size_hidden_layers).to(
            device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=2e-3)
        self.memory = ReplayBuffer(10000)

    def select_action(self, state, current_eps=None):
        sample = random.random()
        if current_eps == None:
            eps_threshold = self.epsilon
        else:
            eps_threshold = current_eps
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # print(policy_net(state),policy_net(state).max(1)[1].view(1, 1))
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0.0
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.

        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        if sum(non_final_mask) > 0:
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
        else:
            non_final_next_states = torch.empty(0, self.state_dim).to(device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)

        with torch.no_grad():
            if sum(non_final_mask) > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            else:
                next_state_values = torch.zeros_like(next_state_values)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute loss
        loss = ((state_action_values - expected_state_action_values.unsqueeze(1)) ** 2).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_steps += 1

        if self.iter_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # epsilon dacay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        return loss.item()


def play_an_dqn_episode(env, agent):
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    state = torch.tensor(state).float().unsqueeze(0).to(device)

    frames_obs = state
    for t in count():
        # Select and perform an action
        action = agent.select_action(frames_obs)
        actions.append(action.item())
        next_state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

        reward = torch.tensor([reward], device=device)
        # Observe new state
        if not done:
            next_state = next_state
            states.append(next_state)
            next_state = torch.tensor(next_state).float().unsqueeze(0).to(device)
            # Move to the next state
            frames_obs = next_state
        else:
            next_state = next_state
            states.append(next_state)
            break
    # print('Total reward for this episode is ', np.sum(rewards))

    env.close()
    return rewards, states, actions


def dqn_train_main(env, NUM_EPISODES=1600):
    dqn_rewards_episode = []
    dqn_loss_episode = []

    state_dim = obs_dim
    n_actions = 2
    # Initialize a dqn agent.
    agent = Agent(env, state_dim, n_actions)
    for i_episode in range(NUM_EPISODES):
        if i_episode % 20 == 0:
            print("episode ", i_episode, "/", NUM_EPISODES)
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        rewards = 0
        frames_obs = state
        loss_temp = []
        for t in range(num_steps):
            # Select and perform an action
            action = agent.select_action(frames_obs)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            rewards += reward

            frames_obs_cur = frames_obs.clone().detach()
            # Observe new state
            if not done:
                next_state = next_state
                next_state = torch.tensor(next_state).float().unsqueeze(0).to(device)
                # Move to the next state
                frames_obs = next_state
            else:
                next_state = None
                frames_obs = None
            # Store the transition in memory
            agent.memory.push(frames_obs_cur, action, frames_obs, reward)
            # Perform one step of the optimization (on the policy network)
            loss = agent.optimize_model()
            loss_temp.append(loss)
            if t == num_steps - 1:
                dqn_rewards_episode.append(rewards.item())

                dqn_loss_episode.append(np.array(loss_temp).mean())
                if random.random() < 0.5:
                    print('episode rewards;', rewards)

                break
    return agent, dqn_rewards_episode, dqn_loss_episode


if __name__ == '__main__':
    env = chain_env(n_state)
    dqn_agent, dqn_rewards_episode, dqn_loss_episode = dqn_train_main(env, num_train_episode)
    plt.figure(1)
    plt.plot(dqn_rewards_episode)
    plt.ylabel('episode reward')
    plt.xlabel('training episode')
    plt.title('DQN agent training episode reward change')

    # plt.savefig('fig/dqn_reward.png')

    plt.figure(2)
    plt.ylabel('episode loss')
    plt.plot(dqn_loss_episode)

    plt.xlabel('training episode')
    plt.title('DQN agent training loss change')
    # plt.savefig('fig/dqn_loss.png')
    plt.show()
    df = pd.DataFrame({'dqn cartpole reward': dqn_rewards_episode, 'dqn chain loss': dqn_loss_episode})
    # df.to_csv('data/dqn_chain.csv')
    # env.close()