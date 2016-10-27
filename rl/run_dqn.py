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
    'RENDER': False,
    'GYM_ENV_NAME': 'CartPole-v0',
    'SOLVED_MEAN_REWARD': 195.0,
    'MAX_STEPS': 200,
    'MAX_EPISODES': 5000,
    'MAX_HISTORY': 100
}

param_range = {
    'gamma': [0.99, 0.95, 0.90],
    'learning_rate': [1., 0.1],
    'e_anneal_steps': [1000, 10000],
    'n_epoch': [1, 2]
}
param_grid = param_product(param_range)


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


if __name__ == '__main__':
    run_session(
        param={'e_anneal_steps': 5000,
               'learning_rate': 0.1,
               'n_epoch': 20,
               'gamma': 0.99})

    # # advanced parallel param selection from util
    # best_param = select_best_param(run_session, sys_vars, param_grid)
    # logger.info(pp.pformat(best_param))
