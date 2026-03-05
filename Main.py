#基于Double DQN解决CartPole_v1

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple,deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("现在使用的设备是："+str(device))

"""Replay Memory_经验回放"""

Transition = namedtuple("Transition",("state","action","next_state","reward"))

class ReplayMemory():   #创建经验池

    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
    
    def push(self,*args):   
        #*解包操作，args是一个包含了Transition元素的元组
        self.memory.append(Transition(*args))   #添加一个transition到memory中
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    """创建Q网络"""

class DQN(nn.Module):

    def __init__(self,n_observations,n_actions):
        super().__init__()

        #创建三层线性神经网络
        self.layer1 = nn.Linear(n_observations,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,n_actions)
        
    def forward(self,x):   #前向传播,relu激活
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        
"""超参数&utilities"""

CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.01      #软更新系数
LR = 3e-4

n_actions = env.action_space.n
state,info = env.reset()
n_observations = len(state)   #获取输入输出num

policy_net = DQN(n_observations,n_actions).to(device)  #创建策略网络
target_net = DQN(n_observations,n_actions).to(device)  #创建目标网络
target_net.load_state_dict(policy_net.state_dict())    #将策略网络初值赋给目标网络

optimizer = optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
memory = ReplayMemory(CAPACITY)

steps_done = 0 #已完成的步数

def select_action(state):   #eps-greedy策略在给定state下进行action select
    global steps_done
    p = random.random()   #创建一个0-1随机因子p
    #eps指数衰减
    eps_threshod = EPS_END + (EPS_START-EPS_END)* math.exp(-1.*steps_done/EPS_DECAY)
    steps_done +=1
    if p > eps_threshod:     #如果p>eps则不进行探索操作，只进行推理
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)   #view是torch的方法，返回tensor
    else:
        return torch.tensor([[env.action_space.sample()]],      #进行探索
                            device=device,dtype=torch.long)

episode_durations = [] 

"""绘图命令"""
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.0005)  # pause a bit so that plots are updated

"""Traning Loop"""
def optimize_model():          #模型训练与优化
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                batch.next_state)),device = device,dtype =  torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                if s is not None])
    state_batch = torch.cat(batch.state)  
    action_batch = torch.cat(batch.action)  
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1,action_batch)  
    next_state_values = torch.zeros(BATCH_SIZE,device=device)

    with torch.no_grad():
        next_state_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)

    expected_state_action_values = (next_state_values * GAMMA)\
                + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_action_values\
                .unsqueeze(1))
    
    #开始优化
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 350

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

plot_durations(show_result=True)
plt.ioff()
plt.show()

# 保存模型
torch.save(policy_net.state_dict(), "cartpole_dqn.pth")

# 创建测试环境
env_test = gym.make("CartPole-v1", render_mode="human")

# 加载模型（如果需要重新创建网络，这里直接使用 policy_net 也可以）
policy_net.eval()  # 确保处于评估模式

state, info = env_test.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for t in count():
    with torch.no_grad():
        action = policy_net(state).max(1).indices.view(1, 1)
    
    observation, reward, terminated, truncated, _ = env_test.step(action.item())
    
    if terminated or truncated:
        break
    
    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    state = next_state

env_test.close()

print('Complete')