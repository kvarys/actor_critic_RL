import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepActorNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim, action_dim, device ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        self.relu = nn.ReLU()

        self.device = device
        self.to(self.device)

    def forward(self, states):
        # x = F.relu(self.fc1(states))
        # x = F.relu(self.fc2(x))
        # action = self.fc3(x)
        #
        # return action

        x = self.relu(self.fc1(states))
        x = self.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Normalize actions between -1 and 1

        return action

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim, action_dim, device):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.input_dims+action_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)  # Concatenate along feature dimension
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_vals = self.fc3(x)
        return q_vals
