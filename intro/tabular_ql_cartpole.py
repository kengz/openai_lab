import gym
import numpy as np
from collections import deque
from tabular_q_learner import QLearner

env = gym.make('CartPole-v0')

MAX_EPISODES = 100
MAX_STEPS = 200
episode_history = deque(maxlen=100)


def pixelate_state_space(env, coarse_grain=10):
    bounds = [env.observation_space.low, env.observation_space.high]
    state_bounds = np.transpose(bounds)
    state_pixels = [np.linspace(*db, num=coarse_grain+1)
                    for db in state_bounds]
    return state_pixels


COARSE_GRAIN = 10
pixel_state_space = pixelate_state_space(env, COARSE_GRAIN)
# dim = total num of pixels
state_dim = COARSE_GRAIN ** env.observation_space.shape[0]
num_actions = env.action_space.n

q_learner = QLearner(state_dim, num_actions)


# use to biject a state space pixel coor to int
def pixelate_states(observation):
    val_space_pairs = list(zip(observation, pixel_state_space))
    pixel_states = [np.digitize(*val_space) for val_space in val_space_pairs]
    return int("".join([str(ps) for ps in pixel_states]))


def run_episode(env, ep):
    total_rewards = 0
    observation = env.reset()
    state = pixelate_states(observation)
    action = q_learner.initializeState(state)

    for t in range(MAX_STEPS):
        env.render()
        observation, reward, done, info = env.step(action)
        state = pixelate_states(observation)
        total_rewards += reward
        if done:
            reward = -200  # normalize reward when done
        action = q_learner.updateModel(state, reward)
        if done:
            break
    return update_history(total_rewards, ep, t)


def update_history(total_rewards, ep, total_t):
    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)
    print('Episode {}'.format(ep))
    print('Finished after {} timesteps'.format(total_t+1))
    print('Reward for this episode: {}'. format(total_rewards))
    print('Average reward for last 100 episodes: {}'.format(mean_rewards))
    return mean_rewards


def run():
    for ep in range(MAX_EPISODES):
        mean_rewards = run_episode(env, ep)
        if mean_rewards >= 195.0:
            print('Environment solved after {} episodes'.format(ep+1))
            break

run()
