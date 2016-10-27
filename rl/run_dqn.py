import util
import gym
from util import *
from collections import deque
from replay_memory import ReplayMemory
from keras_dqn import DQN

import tensorflow as tf

# import util
# import gym
# import json
# import numpy as np
# from util import *
# from collections import deque
# from time import time
# # from sklearn.grid_search import ParameterGrid
# from multiprocessing import cpu_count
# # from joblib import Parallel, delayed
# from replay_memory import ReplayMemory
# # from dqn import DQN
# from keras_dqn import DQN

# rl sys configs, need to implement the keys as shown in util
sys_vars = {
    'RENDER': True,
    'GYM_ENV_NAME': 'CartPole-v0',
    'SOLVED_MEAN_REWARD': 195.0,
    'MAX_STEPS': 200,
    'MAX_EPISODES': 5000,
    'MAX_HISTORY': 100
}
sys_vars.update({
    'epi': 0,  # episode variable
    # total rewards over eoisodes
    'history': deque(maxlen=sys_vars.get('MAX_HISTORY')),
    'mean_rewards': 0,  # mean of history
    'solved': False
})

MODEL_PATH = 'models/dqn.tfl'

param_sets = {
    'gamma': [0.99, 0.95, 0.90],
    'learning_rate': [1., 0.1],
    'e_anneal_steps': [1000, 10000],
    'n_epoch': [1, 2]
}
# (0.96207920792079205, {'e_anneal_steps': 1000, 'learning_rate': 0.1, 'n_epoch': 2, 'gamma': 0.95})
# param_sets = {
#     'gamma': [0.95],
#     'learning_rate': [0.1],
#     'e_anneal_steps': [1000],
#     'n_epoch': [2]
# }
# param_grid = list(ParameterGrid(param_sets))


def run_episode(env, dqn, replay_memory):
    '''run ane episode, return sys_vars'''
    state = env.reset()
    replay_memory.reset_state(state)
    total_rewards = 0

    for t in range(sys_vars.get('MAX_STEPS')):
        if sys_vars.get('RENDER'):
            env.render()

        action = dqn.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_memory.add_exp(action, reward, next_state, done)
        dqn.train(replay_memory)

        state = next_state
        total_rewards += reward
        if done:
            break

    update_history(sys_vars, t, total_rewards)
    return sys_vars


def run_session(dqn_param={}):
    '''run a session of dqn (like a tf session)'''
    check_sys_vars(sys_vars)
    env = gym.make(sys_vars['GYM_ENV_NAME'])
    env_spec = get_env_spec(env)
    replay_memory = ReplayMemory(env_spec)
    sess = tf.Session()
    dqn = DQN(env_spec, sess, **dqn_param)

    for epi in range(sys_vars['MAX_EPISODES']):
        sys_vars['epi'] = epi
        run_episode(env, dqn, replay_memory)
        if sys_vars['solved']:
            break

    return sys_vars


def run_average_session(param={}):
    '''
    run SESSIONS_PER_PARAM sessions for a param
    get the mean param score for them
    '''
    logger.info(
        'Running average session with param = {}'.format(pp.pformat(param)))
    param_score_history = []
    for i in range(SESSIONS_PER_PARAM):
        solved, param_score = run_session(param)
        param_score_history.append(param_score)
        mean_param_score = np.mean(param_score_history)
        if not solved:
            break
    logger.info(
        'Average param score: ' + pp.pformat(
            [mean_param_score, param]))
    return mean_param_score, param


def select_best_param(param_grid):
    '''
    Parameter selection by taking each param in param_grid
    do run_average_session in parallel
    collect (param, mean_param_score) and sort by highest
    '''
    num_cores = cpu_count()
    ranked_params = Parallel(n_jobs=num_cores)(
        delayed(run_average_session)(param) for param in param_grid)
    ranked_params.sort(key=lambda pair: pair[0], reverse=True)

    for pair in ranked_params:
        logger.debug(pp.pformat(list(pair)))
    return ranked_params[0]


if __name__ == '__main__':
    run_session(
        dqn_param={'e_anneal_steps': 5000,
               'learning_rate': 0.1,
               'n_epoch': 20,
               'gamma': 0.99})

    # best_param = select_best_param(param_grid)
    # logger.info(best_param)
