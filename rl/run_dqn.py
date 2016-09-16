import gym
import logging
import tensorflow as tf
import numpy as np
from collections import deque
from time import time
from sklearn.grid_search import ParameterGrid
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from replay_memory import ReplayMemory
from dqn import DQN


SOLVED_MEAN_REWARD = 195.0
MAX_STEPS = 200
MAX_EPISODES = 200
MAX_HISTORY = 100
SESSIONS_PER_PARAM = 5
MODEL_PATH = 'models/dqn.tfl'

# param_sets = {
#     'gamma': [0.99, 0.95, 0.90],
#     'learning_rate': [1., 0.1],
#     'e_anneal_steps': [1000, 10000],
#     'n_epoch': [1, 2]
# }
param_sets = {
    'gamma': [0.99, 0.95],
    'learning_rate': [0.1],
    'e_anneal_steps': [1000],
    'n_epoch': [1]
}
param_grid = list(ParameterGrid(param_sets))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s')

logging.error(len(param_grid))
# logging.info(param_grid)


def get_env_spec(env):
    '''
    Helper: return the env specs: dims, actions
    '''
    return {
        'state_dim': env.observation_space.shape[0],
        'state_bounds': np.transpose(
            [env.observation_space.low, env.observation_space.high]),
        'action_dim': env.action_space.n,
        'actions': list(range(env.action_space.n))
    }


def update_history(epi_history, total_rewards, epi, total_t, epi_time):
    '''
    Helper: update the hisory, max len = MAX_HISTORY
    report status
    return [bool] solved
    '''
    epi_history.append(total_rewards)
    mean_rewards = np.mean(epi_history)
    avg_speed = float(epi_time)/float(total_t)
    logs = [
        '',
        '{:->20}'.format(''),
        'Episode {}'.format(epi),
        'Finished at t={}, reward={}'.format(total_t, total_rewards),
        'Average reward for the last {} episodes: {:.4f}'.format(
            MAX_HISTORY, mean_rewards),
        'Average time per step {:.4f} s/step'.format(avg_speed)
    ]
    solved = mean_rewards >= SOLVED_MEAN_REWARD
    early_exit = bool(
        epi > float(MAX_EPISODES)/2. and mean_rewards < SOLVED_MEAN_REWARD/2.)

    logging.error('\n'.join(logs))
    if solved or (epi == MAX_EPISODES - 1):
        logging.error('Problem solved? {}'.format(solved))
    return mean_rewards, solved, early_exit


def run_episode(epi_history, env, replay_memory, dqn, epi):
    '''
    run an episode
    return [bool] if the problem is solved by this episode
    '''
    total_rewards = 0
    start_time = time()
    state = env.reset()
    replay_memory.reset_state(state)

    for t in range(MAX_STEPS):
        # env.render()
        action = dqn.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_memory.add_exp(action, reward, next_state, done)
        dqn.train(replay_memory)
        state = next_state
        total_rewards += reward
        if done:
            break

    epi_time = time() - start_time
    return update_history(epi_history, total_rewards, epi, t, epi_time)


def run_session(param={}):
    '''
    primary singular method
    '''
    epi_history = deque(maxlen=MAX_HISTORY)
    env = gym.make('CartPole-v0')
    env_spec = get_env_spec(env)
    sess = tf.Session()
    replay_memory = ReplayMemory(env_spec)
    dqn = DQN(env_spec, sess, **param)

    # dqn.restore(MODEL_PATH+'-30')
    for epi in range(MAX_EPISODES):
        mean_rewards, solved, early_exit = run_episode(
            epi_history, env, replay_memory, dqn, epi)
        if solved or early_exit:
            break

    # dqn.save(MODEL_PATH)  # save final model
    param_score = mean_rewards/float(epi)
    return solved, param_score


def run_average_session(param={}):
    '''
    run SESSIONS_PER_PARAM sessions for a param
    get the mean param score for them
    '''
    param_score_history = []
    for i in range(SESSIONS_PER_PARAM):
        solved, param_score = run_session(param)
        param_score_history.append(param_score)
        mean_param_score = np.mean(param_score_history)
        if not solved:
            break
    return param, mean_param_score


def select_best_param(param_grid):
    '''
    Parameter selection by taking each param in param_grid
    do run_average_session in parallel
    collect (param, mean_param_score) and sort by highest
    '''
    num_cores = cpu_count()
    ranked_params = Parallel(n_jobs=num_cores)(
        delayed(run_average_session)(param) for param in param_grid)
    ranked_params.sort(key=lambda pair: pair[1], reverse=True)

    for pair in ranked_params:
        logging.info(pair[0])
        logging.info(pair[1])
    return ranked_params[0]


if __name__ == '__main__':
    select_best_param(param_grid)
