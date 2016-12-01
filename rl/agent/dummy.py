import numpy as np
from rl.agent.base_agent import Agent


class Dummy(Agent):

    '''
    A dummy agent that does random actions, for demo
    '''

    def select_action(self, state):
        '''epsilon-greedy method'''
        action = np.random.choice(self.env_spec['actions'])
        return action

    def update(self, sys_vars):
        return

    def to_train(self, sys_vars):
        return True

    def train(self, sys_vars):
        return
