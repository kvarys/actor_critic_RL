import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actor_critic_RL.networks import DeepActorNetwork, DeepQNetwork
from actor_critic_RL.replay_buffer import ReplayBuffer
from replay_buffer import NStepPrioritizedExperienceReplay, NStepReplayBuffer, ReplayBuf


class DDPG:
    """Deep Deterministic Policy Gradient actor-critic agent"""
    def __init__(self, gamma, lr_critic, lr_actor, input_dims, hidden_dim, action_dim, batch_size, replay_buffer, device, min_buf_count=50000):

        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.time_cntr = 0
        self.online_policy_update_delay = 1
        self.min_buf_count = min_buf_count

        self.replay_buffer = replay_buffer

        self.online_policy = DeepActorNetwork(input_dims=input_dims,
                                              hidden_dim=hidden_dim,
                                              action_dim=action_dim,
                                              device=device)

        self.online_Q_net = DeepQNetwork(input_dims=input_dims,
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

        self.Q_fun_optimizer = optim.Adam(self.online_Q_net.parameters(), lr=lr_critic)
        self.policy_optimizer = optim.Adam(self.online_policy.parameters(), lr=lr_actor)

        # Noise parameters
        self.noise_decay_rate = 0.99999  # Control the decay rate of the noise
        self.noise_stddev = 0.5  # Initial noise standard deviation (controls exploration)

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
        online_q = self.online_Q_net.forward(states, actions) # compute gradients
        target_q = self.compute_Q_fun_target(rewards=rewards, s_=s_, terminals=terminals) # skips gradient computations
        q_loss = self.Q_fun_loss(target_q, online_q).to(self.device)
        return q_loss, online_q, target_q
        # return q_loss, torch.mean(online_q), torch.mean(target_q)

    def compute_actor_loss(self, states):
        '''
        Gradient ascent towards the online_Q_fun
        '''
        return -self.online_Q_net.forward(states=states, actions=self.online_policy.forward(states)).mean()

    def update_online_critic(self,states, actions, rewards, s_, terminals, per=False, index=None, weights=None):
        '''

        '''
        self.Q_fun_optimizer.zero_grad()
        q_loss, online_q, target_q = self.compute_msbe(states=states, actions=actions, rewards=rewards, s_=s_, terminals=terminals)

        if per:
            tderror = torch.abs(online_q[0] - target_q[0])
            self.replay_buffer.update_priority(tderror, index)
            q_loss = torch.mean(q_loss*weights) # TODO je tohle spravne?

        q_loss.backward()
        critic_grad_norm_before = torch.norm(torch.stack([p.grad.norm() for p in self.online_Q_net.parameters() if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(self.online_Q_net.parameters(), max_norm=1.0) # gradient clipping
        critic_grad_norm_after = torch.norm(torch.stack([p.grad.norm() for p in self.online_Q_net.parameters() if p.grad is not None]))
        self.Q_fun_optimizer.step()
        return q_loss, torch.mean(online_q), torch.mean(target_q), critic_grad_norm_before, critic_grad_norm_after

    def _check_whether_to_learn(self):
        '''
        Is the replay buffer full enough to start learning?
        '''
        per = isinstance(self.replay_buffer, NStepPrioritizedExperienceReplay)
        if per:
            if self.replay_buffer.replay_buffer.replay_buffer.mem_cntr < self.min_buf_count:
                return False
        elif isinstance(self.replay_buffer, NStepReplayBuffer):
            if self.replay_buffer.replay_buffer.mem_cntr < self.min_buf_count:
                return False
        elif isinstance(self.replay_buffer, ReplayBuf):
            if self.replay_buffer.mem_cntr < self.min_buf_count:
                return False
        return True

    def return_none(self):
        '''
        When learning is not enabled yet
        '''
        return None, None, None, None, None, None, None

    def learn(self):
        '''
        learn the online policy and online_Q_fun
        '''

        if not self._check_whether_to_learn():
            return self.return_none()

        self.online_policy.train()  # Re-enable training mode

        if isinstance(self.replay_buffer, NStepPrioritizedExperienceReplay):
            (states, actions, rewards, s_, terminals), weights, index = self.replay_buffer.sample_buffer(self.batch_size)
            q_loss, online_q, target_q, critic_grad_norm_before, critic_grad_norm_after = self.update_online_critic(states=states,actions=actions,rewards=rewards,s_=s_,terminals=terminals,per=True,index=index, weights=weights)
        else:
            states, actions, rewards, s_, terminals = self.replay_buffer.sample_buffer(self.batch_size)
            q_loss, online_q, target_q, critic_grad_norm_before, critic_grad_norm_after = self.update_online_critic(states=states,actions=actions,rewards=rewards,s_=s_,terminals=terminals)

        # update the online policy
        actor_loss = None
        actor_grad_norm = None
        if self.time_cntr % self.online_policy_update_delay == 0:
            self.policy_optimizer.zero_grad()
            actor_loss = self.compute_actor_loss(states=states)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.online_policy.parameters(), max_norm=1.0) # gradient clipping
            self.policy_optimizer.step()

        self.time_cntr += 1
        self.update_target_nets() # should we update target networks every time step?

        self.decay_noise()

        return actor_loss, q_loss, online_q, target_q, critic_grad_norm_before, critic_grad_norm_after, actor_grad_norm

    def decay_noise(self):
        """
        Decay the exploration noise linearly over time.
        """
        # Decrease the noise standard deviation linearly over episodes or timesteps
        self.noise_stddev = max(self.noise_stddev * self.noise_decay_rate, 0.01)  # Set a minimum noise level (e.g., 0.01)

    def choose_action(self, state):
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

    def update_target_nets(self, tau=0.005):
        """Soft update target networks: θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(self.target_Q_net.parameters(), self.online_Q_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

        for target_param, online_param in zip(self.target_policy.parameters(), self.online_policy.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

    def save_params(self, name):
        torch.save({
            'actor': self.online_policy.state_dict(),
            'critic': self.online_Q_net.state_dict(),
            'target_actor': self.target_policy.state_dict(),
            'target_critic': self.target_Q_net.state_dict(),
        }, f"{name}_ddpg_weights.pth")

    def load_params(self, name):
        checkpoint = torch.load(f"{name}_ddpg_weights.pth", map_location=self.device)
        self.online_policy.load_state_dict(checkpoint['actor'])
        self.online_Q_net.load_state_dict(checkpoint['critic'])
        self.target_policy.load_state_dict(checkpoint['target_actor'])
        self.target_Q_net.load_state_dict(checkpoint['target_critic'])

class TD3(DDPG):
    """Twin-Delayed DDPG"""
    def __init__(self,min_buf_count, gamma, lr_critic, lr_actor, input_dims, hidden_dim, action_dim, batch_size, replay_buffer, device):

        DDPG.__init__(self, gamma=gamma, lr_critic=lr_critic, lr_actor=lr_actor, input_dims=input_dims, hidden_dim=hidden_dim, action_dim=action_dim, batch_size=batch_size, replay_buffer=replay_buffer, device=device, min_buf_count=min_buf_count)

        self.online_policy_update_delay = 2

        self.online_Q_net_2 = DeepQNetwork(input_dims=input_dims,
                                         hidden_dim=hidden_dim,
                                         action_dim=action_dim,
                                         device=device)

        self.target_Q_net_2 = DeepQNetwork(input_dims=input_dims,
                                         hidden_dim=hidden_dim,
                                         action_dim=action_dim,
                                         device=device)

    def compute_Q_fun_target(self, rewards, s_, terminals):
        '''
        rewards, s_ and terminals are all torch tensors on self.device
        '''
        with torch.no_grad():
            target_actions_next = self.target_policy.forward(s_)

            q_target_1 = self.target_Q_net.forward(states=s_, actions=target_actions_next)
            q_target_1[terminals] = 0.0

            q_target_2 = self.target_Q_net_2.forward(states=s_, actions=target_actions_next)
            q_target_2[terminals] = 0.0

            q_target = rewards + self.gamma * torch.min(q_target_1, q_target_2) * (1 - terminals.float())

        return q_target

    def compute_msbe(self, states, actions, rewards, s_, terminals):
        '''
        Computes the mean-squared Bellman error used to learn the Q function
        '''
        online_q_1 = self.online_Q_net.forward(states, actions) # compute gradients
        online_q_2 = self.online_Q_net_2.forward(states, actions) # compute gradients
        target_q = self.compute_Q_fun_target(rewards=rewards, s_=s_, terminals=terminals) # skips gradient computations
        q_loss_1 = self.Q_fun_loss(target_q, online_q_1).to(self.device)
        q_loss_2 = self.Q_fun_loss(target_q, online_q_2).to(self.device)
        return (q_loss_1, q_loss_2), (torch.mean(online_q_1), torch.mean(online_q_2)), torch.mean(target_q) # online_q_1 will always be used here for per is that correct?

    def update_online_critic(self, states, actions, rewards, s_, terminals, per=False, index=None, weights=None):
        '''

        '''
        self.Q_fun_optimizer.zero_grad()
        (q_loss_1, q_loss_2), online_q, target_q = self.compute_msbe(states=states, actions=actions, rewards=rewards, s_=s_, terminals=terminals)

        if per:
            tderror = torch.abs(online_q[0] - target_q[0])
            self.replay_buffer.update_priority(tderror, index)
            q_loss_1 = torch.mean(q_loss_1*weights) # TODO je tohle spravne?
            q_loss_2 = torch.mean(q_loss_2*weights) # TODO je tohle spravne?

        q_loss_1.backward()
        q_loss_2.backward()
        critic_grad_norm_before_1 = torch.norm(torch.stack([p.grad.norm() for p in self.online_Q_net.parameters() if p.grad is not None]))
        critic_grad_norm_before_2 = torch.norm(torch.stack([p.grad.norm() for p in self.online_Q_net_2.parameters() if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(self.online_Q_net.parameters(), max_norm=1.0) # gradient clipping
        critic_grad_norm_after_1 = torch.norm(torch.stack([p.grad.norm() for p in self.online_Q_net.parameters() if p.grad is not None]))
        critic_grad_norm_after_2 = torch.norm(torch.stack([p.grad.norm() for p in self.online_Q_net_2.parameters() if p.grad is not None]))
        self.Q_fun_optimizer.step()
        return (q_loss_1, q_loss_2), online_q, target_q, (critic_grad_norm_before_1, critic_grad_norm_before_2), (critic_grad_norm_after_1, critic_grad_norm_after_2)

    def update_target_nets(self, tau=0.005):
        """Soft update target networks: θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(self.target_Q_net.parameters(), self.online_Q_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

        for target_param, online_param in zip(self.target_Q_net_2.parameters(), self.online_Q_net_2.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

        for target_param, online_param in zip(self.target_policy.parameters(), self.online_policy.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

    def return_none(self):
        '''
        When learning is not enabled yet
        '''
        return None, (None,None), (None, None),None, (None,None), (None, None), None
