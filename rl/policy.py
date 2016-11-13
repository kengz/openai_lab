import numpy as np


class Policy(object):

    '''
    The base class of Policy, with the core methods
    Acts as a proxy policy definition,
    still draws parameters from agent to compute
    '''

    def __init__(self, agent):
        '''
        call from Agent.__init__ as:
        self.policy = Policy(self)
        '''
        self.agent = agent

    def select_action(self, state):
        raise NotImplementedError()

    def update(self, sys_vars, replay_memory):
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):

    '''
    The Epsilon-greedy policy
    '''

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        agent = self.agent
        epi = sys_vars['epi']
        mem_size = replay_memory.size()
        rise = agent.final_e - agent.init_e
        slope = rise / float(agent.e_anneal_steps)
        agent.e = max(slope * mem_size + agent.init_e, agent.final_e)
        return agent.e

    def select_action(self, state):
        '''epsilon-greedy method'''
        agent = self.agent
        if agent.e > np.random.rand():
            action = np.random.choice(agent.env_spec['actions'])
        else:
            state = np.reshape(state, (1, state.shape[0]))
            Q_state = agent.model.predict(state)
            action = np.argmax(Q_state)
        return action


class OscillatingEpsilonGreedyPolicy(EpsilonGreedyPolicy):

    '''
    The epsilon-greedy policy with oscillating epsilon
    periodically agent.e will drop to a fraction of
    the current exploration rate
    '''

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        super(OscillatingEpsilonGreedyPolicy, self).update(
            sys_vars, replay_memory)
        agent = self.agent
        epi = sys_vars['epi']
        if not (epi % 3) and epi > 15:
            # drop to 1/3 of the current exploration rate
            agent.e = max(agent.e/3., agent.final_e)
        return agent.e
