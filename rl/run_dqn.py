import gym
from util import *
from replay_memory import ReplayMemory
from keras_dqn import DQN


def run_episode(sys_vars, env, dqn, replay_memory):
    '''run ane episode, return sys_vars'''
    state = env.reset()
    replay_memory.reset_state(state)
    total_rewards = 0
    logger.debug("DQN params: e: {} learning_rate: {} "
          "batch size: {} num_epochs: {}".format(
              dqn.e, dqn.learning_rate,
              dqn.batch_size, dqn.n_epoch))

    for t in range(sys_vars.get('MAX_STEPS')):
        if sys_vars.get('RENDER'):
            env.render()

        action = dqn.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_memory.add_exp(action, reward, next_state, done)
        dqn.train(sys_vars, replay_memory)
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
    dqn = DQN(env_spec, **param)

    for epi in range(sys_vars['MAX_EPISODES']):
        sys_vars['epi'] = epi
        run_episode(sys_vars, env, dqn, replay_memory)
        # Best so far, increment num epochs every 2 up to a max of 5
        # TODO: eh? absorb?
        if (dqn.n_epoch < 5 and epi % 2 == 0):
            dqn.n_epoch += 1
        if sys_vars['solved']:
            break

    return sys_vars


if __name__ == '__main__':
    run_session(
        problem='CartPole-v0',
        # problem='MountainCar-v0',
        param={'e_anneal_steps': 2500,
               'learning_rate': 0.01,
               'gamma': 0.95})

    # # advanced parallel param selection from util
    # # for hyper-param selection
    # param_range = {
    #     'gamma': [0.99, 0.95, 0.90],
    #     'learning_rate': [0.01, 0.02, 0.05],
    #     'e_anneal_steps': [2500, 5000]
    # }
    # param_grid = param_product(param_range)

    # best_param = select_best_param(run_session, 'CartPole-v0', param_grid)
    # logger.info(pp.pformat(best_param))
