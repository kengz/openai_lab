import util
import gym
from util import *
from replay_memory import ReplayMemory
from keras_dqn import DQN

import tensorflow as tf  # to be removed
# TODO: implement param grid too


# rl sys configs, need to implement the required_sys_keys in util
# only implement constants here,
# on reset will add vars: {epi, history, mean_rewards, solved}
sys_vars = {
    'RENDER': True,
    'GYM_ENV_NAME': 'CartPole-v0',
    'SOLVED_MEAN_REWARD': 195.0,
    'MAX_STEPS': 200,
    'MAX_EPISODES': 5000,
    'MAX_HISTORY': 100
}

SESSIONS_PER_PARAM = 10
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


def run_session(param={}):
    '''run a session of dqn (like a tf session)'''
    reset_sys_vars(sys_vars)  # reset sys_vars per session
    env = gym.make(sys_vars['GYM_ENV_NAME'])
    env_spec = get_env_spec(env)
    replay_memory = ReplayMemory(env_spec)
    sess = tf.Session()
    dqn = DQN(env_spec, sess, **param)

    for epi in range(sys_vars['MAX_EPISODES']):
        sys_vars['epi'] = epi
        run_episode(env, dqn, replay_memory)
        if sys_vars['solved']:
            break

    return sys_vars


def run_session_average(param={}):
    '''
    run session multiple times for a param
    then average the mean_rewards from them
    '''
    logger.info(
        'Running average session with param = {}'.format(pp.pformat(param)))
    mean_rewards_history = []
    for i in range(SESSIONS_PER_PARAM):
        run_session(param)
        mean_rewards_history.append(sys_vars['mean_rewards'])
        sessions_mean_rewards = np.mean(mean_rewards_history)
        if sys_vars['solved']:
            break
    logger.info(
        'Sessions mean rewards: {}'.format(sessions_mean_rewards))
    return {'param': param, 'sessions_mean_rewards': sessions_mean_rewards}


def select_best_param(param_grid):
    '''
    Parameter selection 
    by running session average for each param parallel
    then sort by highest sessions_mean_rewards first
    return the best
    '''
    NUM_CORES = multiprocessing.cpu_count()
    p = multiprocessing.Pool(NUM_CORES)
    params_means = p.map(run_session_average, param_grid)
    params_means.sort(key=lambda pm: pm['sessions_mean_rewards'], reverse=True)
    for pm in params_means:
        logger.debug(pp.pformat(pm))
    return params_means[0]


if __name__ == '__main__':
    run_session(
        param={'e_anneal_steps': 5000,
               'learning_rate': 0.1,
               'n_epoch': 20,
               'gamma': 0.99})

    # best_param = select_best_param(param_grid)
    # logger.info(pp.pformat(best_param))
