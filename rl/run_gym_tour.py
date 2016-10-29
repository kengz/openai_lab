# A tour of the gym
import gym
from util import *
from replay_memory import ReplayMemory
from keras_dqn import DQN

# rl sys variables - see util.py for info
sys_vars = init_sys_vars('CartPole-v0')


def run_episode(env, dqn, replay_memory):
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
        # dqn.train(replay_memory)

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
    dqn = 'DUMMY'

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
