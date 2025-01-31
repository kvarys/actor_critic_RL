from agent import DQN, C51NoisyNstep3DQN, DoubleDQN, DDDQN, NStep3DQN, NoisyNStep3DQN
import numpy as np
import time
from replay_buffer import ReplayBuffer
import torch as T
import gymnasium as gym
import os
import argparse
from utils import *


def make_env(game):
    return gym.make("ALE/" + game + "-ram-v5")

def initialize_agent():
    GAMMA = 0.99
    MAX_REPLAY_SIZE = 1048576
    EPSILON = 1
    DEVICE = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    replay_buffer = ReplayBuffer(max_size=MAX_REPLAY_SIZE, input_shape=env.observation_space.shape[0], device=DEVICE)

    if agent_name=="DQN":
        return DQN(gamma=GAMMA, epsilon=EPSILON, lr=lr,input_dims=env.observation_space.shape[0],batch_size=batch_size,
                   n_actions=env.action_space.n,max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_target=update_target, eps_steps=eps_steps, eps_min=eps_min)
    elif agent_name=="DDQN":
        return DoubleDQN(gamma=GAMMA, epsilon=EPSILON, lr=lr,input_dims=env.observation_space.shape[0],batch_size=batch_size,
                         n_actions=env.action_space.n,max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_target=update_target, eps_steps=eps_steps, eps_min=eps_min)
    elif agent_name=="DDDQN":
        return DDDQN(gamma=GAMMA, epsilon=EPSILON, lr=lr, input_dims=env.observation_space.shape[0], batch_size=batch_size,
                     n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_target=update_target, eps_steps=eps_steps, eps_min=eps_min)
    elif agent_name=="NStep3DQN":
        return NStep3DQN(gamma=GAMMA, epsilon=EPSILON, lr=lr, input_dims=env.observation_space.shape[0], batch_size=batch_size,
                         n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_target=update_target, eps_steps=eps_steps, eps_min=eps_min)
    elif agent_name=="NoisyNStep3DQN":
        return NoisyNStep3DQN(gamma=GAMMA, epsilon=EPSILON, lr=lr, input_dims=env.observation_space.shape[0], batch_size=batch_size,replay_buffer=prioritized_replay_buffer, n_actions=env.action_space.n, max_mem_size=MAX_REPLAY_SIZE, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_target=update_target, eps_steps=eps_steps, eps_min=eps_min)
    elif agent_name=="C51NoisyNStep3DQN":
        return C51NoisyNstep3DQN(bins=51, min_support=-10, max_support=10, gamma=GAMMA, epsilon=EPSILON, lr=lr, input_dims=env.observation_space.shape[0], batch_size=batch_size,replay_buffer=prioritized_replay_buffer, n_actions=env.action_space.n, max_mem_size=MAX_REPLAY_SIZE, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_target=update_target, eps_steps=eps_steps, eps_min=eps_min)
    return None

if __name__ == '__main__':

    AGENT_NAME = "NONE"

    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str)

    parser.add_argument('--fc1', type=int, default=256)
    parser.add_argument('--fc2', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--update_target', type=int, default=100)
    parser.add_argument('--agent_name', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eps_steps', type=int, default=100000)
    parser.add_argument('--eps_min', type=float, default=0.01)

    args = parser.parse_args()

    game = args.game
    print("Playing Game: " + str(game))

    agent_name = args.agent_name
    fc1_dims = args.fc1
    fc2_dims = args.fc2
    update_target = args.update_target
    lr = args.lr
    batch_size = args.batch_size
    eps_steps = args.eps_steps
    eps_min = args.eps_min

    env = make_env(game)

    agent = initialize_agent()

    AGENT_NAME = agent_name+"_"+non_default_args(args, parser)

    n_steps = 4000000
    steps = 0
    done = False
    observation, info = env.reset()
    last_time = time.time()
    score = 0
    episode_scores = []
    steps_per_episode = []
    episodes = 0
    last_steps = 1

    while steps < n_steps:

        action = agent.choose_action(observation)

        observation_, reward, done_, trun_, info = env.step(action)
        done_ = np.logical_or(done_, trun_)
        steps += 1

        score += reward

        agent.learn()

        reward = np.clip(reward, -1., 1.)

        agent.store_transition(observation, action, reward, observation_, done_)

        observation = observation_

        if done_:
            episode_scores.append(score)
            steps_per_episode.append(steps)
            score = 0
            env.reset()

        if steps % 1200 == 0 and len(episode_scores) > 0:

            avg_score = np.mean(episode_scores[-50:])

            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'.format(AGENT_NAME, game, avg_score, steps,
                                                        (steps - last_steps) / (time.time() - last_time)), flush=True)
                last_steps = steps
                last_time = time.time()

    print("Finished!")
    episode_scores = np.array(episode_scores)
    steps_per_episode = np.array(steps_per_episode)
    results_combined = np.column_stack((episode_scores, steps_per_episode))
    np.save(AGENT_NAME + ".npy", results_combined)
