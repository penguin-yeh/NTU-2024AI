import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import YOUR_CODE_HERE
import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        
        # build your own CNN model
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layers
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 512) 
        self.fc2 = nn.Linear(512, action_dim)
        
        # utils.raiseNotDefined()
        
        # this is just an example, you can modify this.
        # self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)

    def forward(self, x):
        
        x = F.relu(self.conv1(x)) 
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class ReplayBuffer:
    # referenced [TD3 official implementation](https://github.com/sfujim/TD3/blob/master/utils.py#L5).
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=5e-4,
        epsilon=0.9,
        epsilon_min=0.05,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        """
        DQN agent has four methods.

        - __init__() as usual
        - act() takes as input one state of np.ndarray and output actions by following epsilon-greedy policy.
        - process() method takes one transition as input and define what the agent do for each step.
        - learn() method samples a mini-batch from replay buffer and train q-network
        """
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # output actions by following epsilon-greedy policy
            
            # convert x to tensor, and put it to GPU
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            
            # "*** YOUR CODE HERE ***"
            
            # get q-values from network
            q_value = self.network(x)
            # get action with maximum q-value
            action = q_value.argmax().item()
        
        return action
    
    def learn(self):
        # "*** YOUR CODE HERE ***"
        
        # sample a mini-batch from replay buffer
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        # get action-corresponding q-values from network
        q_values = self.network(state).gather(1, action.long())
        
        # get next q-values from network
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1) # get max q-values from each dim-1
            
        # td_target: if terminated, only reward, otherwise reward + gamma * max(next_q)
        td_target = reward + (1 - terminated) * self.gamma * next_q_values
        
        # compute loss with td_target and q-values
        loss = F.smooth_l1_loss(q_values, td_target)
        
        # initialize optimizer
        self.optimizer.zero_grad()
        
        # backpropagation
        loss.backward()
        
        # update network
        self.optimizer.step()
        
        return {"value_loss": loss.item()} # return dictionary for logging
    
    # transition：state, action, reward, next_state, terminated
    def process(self, transition):
        # "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        
        result = {}
        self.total_steps += 1
        
        # update replay buffer
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            # update target network
            self.target_network.load_state_dict(self.network.state_dict())
        
        # self.epsilon -= self.epsilon_decay
        self.epsilon *= 0.97
        
        return result