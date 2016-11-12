import numpy as np


class Dummy(object):

    '''
    A dummy agent that does random actions, for demo
    '''

    def __init__(self, env_spec):
        self.env_spec = env_spec

    def select_action(self, state):
        '''epsilon-greedy method'''
        action = np.random.choice(self.env_spec['actions'])
        return action

    def train(self, sys_vars, replay_memory):
        return
