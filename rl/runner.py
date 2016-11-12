import gym
from util import *
from replay_memory import ReplayMemory


class Runner(object):

    '''
    The base class for running a session of
    a DQN Agent, at a problem, with agent params
    '''

    def __init__(self, Agent, problem, param):
        self.Agent = Agent
        self.problem = problem
        self.param = param

    def run_episode(self, sys_vars, env, agent, replay_memory):
        '''run ane episode, return sys_vars'''
        state = env.reset()
        replay_memory.reset_state(state)
        total_rewards = 0
        logger.debug("DQN Agent param: e: {} learning_rate: {} "
                     "batch size: {} num_epochs: {}".format(
                         agent.e, agent.learning_rate,
                         agent.batch_size, agent.n_epoch))

        for t in range(sys_vars.get('MAX_STEPS')):
            sys_vars['t'] = t  # update sys_vars t
            if sys_vars.get('RENDER'):
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_memory.add_exp(action, reward, next_state, done)
            agent.train(sys_vars, replay_memory)
            state = next_state
            total_rewards += reward
            if done:
                break

        update_history(sys_vars, t, total_rewards)
        return sys_vars

    def run_session(self):
        '''run a session of agent'''
        sys_vars = init_sys_vars(
            self.problem, self.param)  # rl system, see util.py
        env = gym.make(sys_vars['GYM_ENV_NAME'])
        env_spec = get_env_spec(env)
        replay_memory = ReplayMemory(env_spec)
        agent = self.Agent(env_spec, **self.param)

        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            self.run_episode(sys_vars, env, agent, replay_memory)
            # Best so far, increment num epochs every 2 up to a max of 5
            if sys_vars['solved']:
                break

        return sys_vars

    # def run_param_selection_sessions(self):
    #     # advanced parallel param selection from util
    #     # for hyper-param selection
    #     param_range = {
    #         'gamma': [0.99, 0.95, 0.90],
    #         'learning_rate': [0.01, 0.02, 0.05],
    #         'e_anneal_steps': [2500, 5000]
    #     }
    #     param_grid = param_product(param_range)

    #     best_param = select_best_param(
    #         self.run_session, 'CartPole-v0', param_grid)
    #     logger.info(pp.pformat(best_param))
