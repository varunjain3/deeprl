from env import Environment1
from dqn import train_dqn, NeuralNetwork
from tools import ReplayMemory

import pandas as pd
import numpy as np
import copy
from collections import namedtuple
import time


from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.append('./stockdqn')

data = pd.read_csv('./stockdqn/HDFC.NS.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
print(f"Data Min:{data.index.min()} \t Max: {data.index.max()}")

date_split = '2015-01-01'
train = data[:date_split]
test = data[date_split:]
print(f"Train len: {len(train)} \t Test len: {len(test)}")

env = Environment1(train)
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple(
    'Transition', ('q_state', 'action', 'next_state', 'reward', 'done'))

model_path = None  # f"model_1614470115.pt"

state_size = env.history_t+1

Q = NeuralNetwork(input_size=state_size, output_size=3)
if model_path is not None:
    Q.load_state_dict(torch.load(model_path))
    print(f"Loaded model saved at: {model_path}")

Q_ast = copy.deepcopy(Q)
Q = Q.to(device)
Q_ast = Q_ast.to(device)

optimizer = optim.Adam(Q.parameters(), lr=1e-3)
criterion = nn.MSELoss()

env = Environment1(train)

epoch_num = 100
step_max = len(env.data)-1
memory_size = 2000
batch_size = 128

epsilon = 1.0
epsilon_decrease = 1.e-2
epsilon_min = 0.1
start_reduce_epsilon = 200

save_freq = 5
update_q_freq = 20
gamma = 0.97
show_log_freq = 1

memory = ReplayMemory(memory_size)
total_step = 0
total_rewards = []
total_losses = []

done = False

for epoch in range(epoch_num):
    start = time.time()
    state = env.reset()
    state = torch.reshape(state, [1, state_size])
    total_loss = 0
    total_reward = 0

    for t in tqdm(range(step_max)):

        if np.random.rand() <= epsilon:
            action = torch.randint(3, [], device=device)
        else:
            q_state = Q_ast(state)
            action = torch.argmax(q_state)

        next_state, reward, done = env.step(action)
        next_state = torch.reshape(next_state, [1, state_size])
        total_reward += reward

        memory.push(state, action.reshape(-1, 1), next_state, reward, done)
        state = next_state

        if len(memory) > batch_size:
            trans = memory.sample(batch_size)
            batch = Transition(*zip(*trans))

            state_batch = torch.cat(batch.q_state)
            action_batch = torch.cat(batch.action).flatten()
            reward_batch = torch.FloatTensor(batch.reward).to(device)
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done)

            q_state_batch = Q(state_batch)
            maxq = torch.amax(Q_ast(next_state_batch), axis=1).data
            target = copy.deepcopy(q_state_batch.data)
            target[:, action_batch] = (reward_batch + gamma*maxq*(not done))

            optimizer.zero_grad()
            loss = criterion(q_state_batch, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.data

        total_step += 1
        if total_step % update_q_freq == 0:
            Q_ast = copy.deepcopy(Q)

    if epsilon > epsilon_min and total_step > start_reduce_epsilon:
        epsilon -= epsilon_decrease

    total_losses.append(float(total_loss))
    total_rewards.append(total_reward)

    if(epoch+1) % save_freq == 0:
        torch.save(Q.state_dict(),
                   f"model_{int(time.time())}+epoch{epoch+1:3d}.pt")

    if (epoch+1) % show_log_freq == 0:
        log_reward = sum(
            total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
        log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
        elapsed_time = time.time()-start
        print('\t'.join(
            map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
        start = time.time()

    # print("hello")

torch.save(Q.state_dict(), f"model_{int(time.time())}.pt")


def plot_loss_reward(total_losses, total_rewards):

    figure = tools.make_subplots(rows=1, cols=2, subplot_titles=(
        'loss', 'reward'), print_grid=False)
    figure.append_trace(Scatter(y=total_losses, mode='lines',
                                line=dict(color='skyblue')), 1, 1)
    figure.append_trace(
        Scatter(y=total_rewards, mode='lines', line=dict(color='orange')), 1, 2)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400, width=900, showlegend=False)
    iplot(figure)


plot_loss_reward(total_losses, total_rewards)
