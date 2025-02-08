import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actor_critic_RL.networks import DeepActorNetwork, DeepQNetwork
from actor_critic_RL.replay_buffer import ReplayBuffer
from replay_buffer import NStepPrioritizedExperienceReplay


class DDPG:
    """Deep Deterministic Policy Gradient actor-critic agent"""
    def __init__(self, gamma, lr_critic, lr_actor, input_dims, hidden_dim, action_dim, batch_size, replay_buffer, update_target, device):

        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.update_target = update_target
        self.target_cntr = 0

        self.replay_buffer = replay_buffer

        self.online_policy = DeepActorNetwork(input_dims=input_dims,
                                              hidden_dim=hidden_dim,
                                              action_dim=action_dim,
                                              device=device)

        self.online_Q_fun = DeepQNetwork(input_dims=input_dims,
                                         hidden_dim=hidden_dim,
                                         action_dim=action_dim,
                                         device=device)

        self.target_policy = DeepActorNetwork(input_dims=input_dims,
                                              hidden_dim=hidden_dim,
                                              action_dim=action_dim,
                                              device=device)

        self.target_Q_net = DeepQNetwork(input_dims=input_dims,
                                         hidden_dim=hidden_dim,
                                         action_dim=action_dim,
                                         device=device)

        self.Q_fun_loss = nn.MSELoss()

        self.Q_fun_optimizer = optim.Adam(self.online_Q_fun.parameters(), lr=lr_critic)
        self.policy_optimizer = optim.Adam(self.online_policy.parameters(), lr=lr_actor)

        # Noise parameters
        self.noise_decay_rate = 0.99999  # Control the decay rate of the noise
        self.noise_stddev = 0.3  # Initial noise standard deviation (controls exploration)

    def compute_Q_fun_target(self, rewards, s_, terminals):
        '''
        rewards, s_ and terminals are all torch tensors on self.device
        '''
        with torch.no_grad():
            target_actions_next = self.target_policy.forward(s_) # dims=
            q_target_max_next = self.target_Q_net.forward(states=s_, actions=target_actions_next)
            q_target_max_next[terminals] = 0.0

            q_target = rewards + self.gamma * q_target_max_next * (1 - terminals.float())

        return q_target

    def compute_msbe(self, states, actions, rewards, s_, terminals):
        '''
        Computes the mean-squared Bellman error used to learn the Q function
        '''
        online_q = self.online_Q_fun.forward(states, actions) # compute gradients
        target_q = self.compute_Q_fun_target(rewards=rewards, s_=s_, terminals=terminals) # skips gradient computations
        q_loss = self.Q_fun_loss(target_q, online_q).to(self.device)
        return q_loss, online_q, target_q
        return q_loss, torch.mean(online_q), torch.mean(target_q)

    def compute_actor_loss(self, states):
        '''
        Gradient ascent towards the online_Q_fun
        '''
        return -self.online_Q_fun.forward(states=states, actions=self.online_policy.forward(states)).mean()

        actor_target = self.online_Q_fun.forward(
                states=states,
                actions=self.online_policy.forward(states)
        )
        return -actor_target.mean()

        actions_pred = self.online_policy.forward(states)
        actor_target = self.online_Q_fun.forward(states=states, actions=actions_pred)
        return -actor_target.mean()

    def learn(self):
        '''
        learn the online policy and online_Q_fun
        '''

        if self.replay_buffer.replay_buffer.replay_buffer.mem_cntr < 50000:
            return None, None, None, None, None, None

        self.online_policy.train()  # Re-enable training mode

        # TODO pak smaz
        # neco = self.replay_buffer.sample_buffer(self.batch_size)
        # print(f"neco {neco}")
        # states, actions, rewards, s_, terminals = self.replay_buffer.sample_buffer(self.batch_size)

        per = isinstance(self.replay_buffer, NStepPrioritizedExperienceReplay)
        if per:
            (states, actions, rewards, s_, terminals), weights, index = self.replay_buffer.sample_buffer(self.batch_size)
        else:
            states, actions, rewards, s_, terminals = self.replay_buffer.sample_buffer(self.batch_size)

        # update the online critic
        self.Q_fun_optimizer.zero_grad()
        q_loss, online_q, target_q = self.compute_msbe(states=states, actions=actions, rewards=rewards, s_=s_, terminals=terminals)

        if per:
            tderror = torch.abs(online_q[0] - target_q[0])
            self.replay_buffer.update_priority(tderror, index)

        q_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.online_Q_fun.parameters(), max_norm=1.0) # gradient clipping
        self.Q_fun_optimizer.step()

        # update the online policy
        self.policy_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(states=states)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.online_policy.parameters(), max_norm=1.0) # gradient clipping
        self.policy_optimizer.step()
        # print(f"q_loss {q_loss} actor_loss {actor_loss}")

        self.target_cntr += 1
        if self.target_cntr % self.update_target == 0:
            self.update_target_nets()

        self.decay_noise()

        return actor_loss, q_loss, torch.mean(online_q), torch.mean(target_q), critic_grad_norm, actor_grad_norm

    def decay_noise(self):
        """
        Decay the exploration noise linearly over time.
        """
        # Decrease the noise standard deviation linearly over episodes or timesteps
        self.noise_stddev = max(self.noise_stddev * self.noise_decay_rate, 0.05)  # Set a minimum noise level (e.g., 0.01)

    def choose_action(self, state, noise_std=0.5):
        """
        Selects an action given a state, adding Gaussian noise for exploration.

        Args:
            state (numpy.ndarray): The current state of the environment.
            noise_std (float): Standard deviation of the Gaussian noise.

        Returns:
            numpy.ndarray: The selected action with added exploration noise.
        """
        self.online_policy.eval()  # Disable Dropout and BatchNorm updates

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():  # Disable gradient tracking for efficiency
            action = self.online_policy.forward(state_tensor)

        self.online_policy.train()  # Re-enable training mode

        # Add Gaussian exploration noise
        noise = self.noise_stddev * torch.randn_like(action)  # Generates noise with mean=0, std=noise_std
        action = action + noise

        # Clip action within valid range
        # action = np.clip(action, -1, 1)
        action = torch.clamp(action, min=-1, max=1)

        return action

    def update_target_nets(self, tau=0.001):
        """Soft update target networks: θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(self.target_Q_net.parameters(), self.online_Q_fun.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

        for target_param, online_param in zip(self.target_policy.parameters(), self.online_policy.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

    def save_params(self, name):
        torch.save({
            'actor': self.online_policy.state_dict(),
            'critic': self.online_Q_fun.state_dict(),
            'target_actor': self.target_policy.state_dict(),
            'target_critic': self.target_Q_net.state_dict(),
        }, f"{name}_ddpg_weights.pth")

    def load_params(self, name):
        checkpoint = torch.load(f"{name}_ddpg_weights.pth", map_location=self.device)
        self.online_policy.load_state_dict(checkpoint['actor'])
        self.online_Q_fun.load_state_dict(checkpoint['critic'])
        self.target_policy.load_state_dict(checkpoint['target_actor'])
        self.target_Q_net.load_state_dict(checkpoint['target_critic'])

