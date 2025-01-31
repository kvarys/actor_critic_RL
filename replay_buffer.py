import numpy as np
from collections import deque
import torch as T


class ReplayBuffer(object):
    def __init__(self, minimal_sample_size, max_size, input_shape, device):
        self.mem_size = max_size
        self.minimal_sample_size = minimal_sample_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.device = device
        self.STATE_NORMALIZATION = 1

    def get_state_normalization(self):
        return self.STATE_NORMALIZATION

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state / self.STATE_NORMALIZATION
        self.new_state_memory[index] = state_ / self.STATE_NORMALIZATION

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = T.tensor(self.state_memory[batch]).to(self.device)
        actions = T.tensor(self.action_memory[batch]).to(self.device)
        rewards = T.tensor(self.reward_memory[batch]).to(self.device)
        states_ = T.tensor(self.new_state_memory[batch]).to(self.device)
        terminal = T.tensor(self.terminal_memory[batch]).to(self.device)
        return states, actions, rewards, states_, terminal
