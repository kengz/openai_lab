import gym
import json
import multiprocessing as mp
from functools import partial
from keras import backend as K
from rl.spec import game_specs
from rl.util import *


class Session(object):

    '''
    main.py calls this
    The base class for running a session of
    a DQN Agent, at a problem, with agent params
    '''

    def __init__(self, problem, Agent, Memory, Policy, param):
        self.problem = problem
        self.Agent = Agent
        self.Memory = Memory
        self.Policy = Policy
        self.param = param

    def run_episode(self, sys_vars, env, agent):
        '''run ane episode, return sys_vars'''
        state = env.reset()
        agent.memory.reset_state(state)
        total_rewards = 0
        debug_agent_info(agent)

        for t in range(agent.env_spec['timestep_limit']):
            sys_vars['t'] = t  # update sys_vars t
            if sys_vars.get('RENDER'):
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.add_exp(action, reward, next_state, done)
            agent.update(sys_vars)
            if agent.to_train(sys_vars):
                agent.train(sys_vars)

            state = next_state
            total_rewards += reward
            if done:
                break
        update_history(agent, sys_vars, t, total_rewards)
        return sys_vars

    def run(self):
        '''run a session of agent'''
        sys_vars = init_sys_vars(
            self.problem, self.param)  # rl system, see util.py
        env = gym.make(sys_vars['GYM_ENV_NAME'])
        agent = self.Agent(get_env_spec(env), **self.param)
        memory = self.Memory(**self.param)
        policy = self.Policy(**self.param)
        agent.compile(memory, policy)
        logger.info('Compiled Agent, Memory, Policy')

        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            self.run_episode(sys_vars, env, agent)
            if 'epi_change_learning_rate' in self.param and epi == self.param['epi_change_learning_rate']:
                agent.recompile_model(self.param['learning_rate'] / 10.0)
            if sys_vars['solved']:
                break

        K.clear_session()  # manual gc to fix TF issue 3388
        return sys_vars
