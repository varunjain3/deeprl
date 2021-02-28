# DQN
import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time

from tools import ReplayMemory

from collections import namedtuple
Transition = namedtuple(
    'Transition', ('q_state', 'action', 'next_state', 'reward', 'done'))


def train_dqn(env):

    class Q_Network(nn.Module):

        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            return torch.Tensor(self.model(x)).cuda()

    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    Q = Q.cuda()
    Q_ast = Q_ast.cuda()
    learning_rate = 1e-4
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    epoch_num = 50
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 20
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 1

    memory = ReplayMemory(memory_size)
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # select act
            pact = torch.randint(0, 3, [1]).cuda()
            if np.random.rand() > epsilon:
                pact = Q(torch.Tensor(
                    np.array(pobs, dtype=np.float32).reshape(1, -1)).cuda())
                pact = torch.argmax(pact.data).cuda()

            # act
            obs, reward, done = env.step(pact)

            Transition = namedtuple('Transition',
                                    ('state', 'action', 'next_state', 'reward'))
            # add memory
            memory.push(pobs, pact, obs, reward)

            # train or update q
            if len(memory) > batch_size:
                if total_step % train_freq == 0:
                    shuffled_memory = memory.sample(batch_size)
                    batch = Transition(*zip(*shuffled_memory))

                    state_batch = torch.cat(batch.state)
                    state_action = torch.cat(batch.action)
                    state_reward = torch.cat(batch.reward)
                    q = Q(state_batch)

                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = torch.Tensor(np.array(batch[:, 0].tolist(
                        ), dtype=np.float32).reshape(batch_size, -1)).cuda()
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = torch.Tensor(np.array(
                            batch[:, 2].tolist(), dtype=np.int32)).cuda()
                        b_obs = torch.Tensor(np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(
                            batch_size, -1)).cuda()
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        # b_pobs = torch.Tensor(b_pobs).cuda()
                        q = Q(b_pobs)
                        maxq = torch.max(Q_ast(b_obs), axis=1)
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j] + \
                                gamma*maxq.values[j]*(not b_done[j])
                        optimizer.zero_grad()
                        output = loss(q, target)
                        total_loss += output.data
                        output.backward()
                        optimizer.step()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(
                total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(
                total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('\t'.join(
                map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()

    return Q, total_losses, total_rewards


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.n_actions = output_size
        self.hidden_size = 100
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bottle = nn.Linear(self.hidden_size, 3)
        self.fc3 = nn.Linear(3, self.n_actions)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bottle(x)
        x = F.relu(x)
        y = self.fc3(x)

        return y
