import gym
from util import *
from replay_memory import ReplayMemory
from q_table import QTable


def run_episode(sys_vars, env, qtable, replay_memory):
    '''run ane episode, return sys_vars'''
    state = env.reset()
    replay_memory.reset_state(state)
    total_rewards = 0
    print("qtable params: e: {} learning_rate: {}".format(
        qtable.e, qtable.learning_rate))

    for t in range(sys_vars.get('MAX_STEPS')):
        if sys_vars.get('RENDER'):
            env.render()

        action = qtable.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_memory.add_exp(action, reward, next_state, done)
        qtable.train(replay_memory)
        state = next_state
        total_rewards += reward
        if done:
            break

    update_history(sys_vars, t, total_rewards)
    return sys_vars


def run_session(problem, param={}):
    '''run a session of qtable'''
    sys_vars = init_sys_vars(problem, param)  # rl system, see util.py
    env = gym.make(sys_vars['GYM_ENV_NAME'])
    env_spec = get_env_spec(env)
    replay_memory = ReplayMemory(env_spec)
    qtable = QTable(env_spec, **param)

    for epi in range(sys_vars['MAX_EPISODES']):
        sys_vars['epi'] = epi
        run_episode(sys_vars, env, qtable, replay_memory)
        # Best so far, increment num epochs every 2 up to a max of 5
        if sys_vars['solved']:
            break

    return sys_vars


if __name__ == '__main__':
    run_session(
        problem='CartPole-v0',
        param={'e_anneal_steps': 5000,
               'learning_rate': 0.01,
               'gamma': 0.95})

    # # advanced parallel param selection from util
    # # for hyper-param selection
    # param_range = {
    #     'gamma': [0.99, 0.95, 0.90],
    #     'learning_rate': [1., 0.1],
    #     'e_anneal_steps': [1000, 10000],
    #     'n_epoch': [1, 2]
    # }
    # param_grid = param_product(param_range)

    # best_param = select_best_param(run_session, sys_vars, param_grid)
    # logger.info(pp.pformat(best_param))
