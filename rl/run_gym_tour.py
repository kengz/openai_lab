# A tour of the gym
import gym
from util import *
from replay_memory import ReplayMemory
from keras_dqn import DQN


def run_episode(sys_vars, env, dqn, replay_memory):
    '''run ane episode, return sys_vars'''
    state = env.reset()
    replay_memory.reset_state(state)
    total_rewards = 0

    for t in range(sys_vars.get('MAX_STEPS')):
        if sys_vars.get('RENDER'):
            env.render()

        action = env.action_space.sample()
        # action = dqn.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_memory.add_exp(action, reward, next_state, done)
        # dqn.train(sys_vars, replay_memory)

        state = next_state
        total_rewards += reward
        if done:
            break

    update_history(sys_vars, t, total_rewards)
    return sys_vars


def run_session(problem, param={}):
    '''run a session of dqn'''
    sys_vars = init_sys_vars(problem, param)  # rl system, see util.py
    env = gym.make(sys_vars['GYM_ENV_NAME'])
    env_spec = get_env_spec(env)
    replay_memory = ReplayMemory(env_spec)
    dqn = 'DUMMY'

    for epi in range(sys_vars['MAX_EPISODES']):
        sys_vars['epi'] = epi
        run_episode(sys_vars, env, dqn, replay_memory)
        if sys_vars['solved']:
            break

    return sys_vars


if __name__ == '__main__':
    run_session(
        problem='CartPole-v0',
        param={'e_anneal_steps': 5000,
               'learning_rate': 0.1,
               'n_epoch': 20,
               'gamma': 0.99})
