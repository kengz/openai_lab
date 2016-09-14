import numpy as np
from collections import deque


class ReplayMemory(object):

    '''
    The replay memory used for random minibatch training
    '''

    def __init__(self, env_spec):
        self.env_spec = env_spec
        self.memory = []
        self.state = None

    def reset_state(self, init_state):
        '''
        reset the state of ReplayMemory per episode env.reset()
        '''
        self.state = init_state

    def add_exp(self, action, reward, next_state, terminal):
        '''
        after the env.step(a) that returns s', r,
        using the previously stored state for the s,
        form an experience tuple <s, a, r, s'>
        '''
        exp = dict(zip(
            ['state', 'action', 'reward', 'next_state', 'terminal'],
            [self.state, action, reward, next_state, terminal]))
        # store exp, update state and time
        self.memory.append(exp)
        self.state = next_state
        return exp

    def get_exp(self, index):
        return self.memory[index]

    def size(self):
        return len(self.memory)

    def one_hot_action(self, action):
        action_arr = np.zeros(self.env_spec['action_dim'])
        action_arr[action] = 1
        return action_arr

    def format_minibatch(self, exp_batch):
        '''
        transpose, transform the minibatch into useful form
        '''
        minibatch = dict(zip(
            ['states', 'actions', 'rewards', 'next_states', 'terminals'],
            [
                np.array([exp['state'] for exp in exp_batch]),
                np.array([self.one_hot_action(exp['action'])
                          for exp in exp_batch]),
                np.array([exp['reward'] for exp in exp_batch]),
                np.array([exp['next_state'] for exp in exp_batch]),
                np.array([exp['terminal'] for exp in exp_batch])
            ]
        ))
        return minibatch

    def rand_minibatch(self, size):
        '''
        get a minibatch of random exp for training
        '''
        memory_size = self.size()
        if memory_size <= size:
            # to prevent repetition and initial overfitting
            rand_inds = np.random.permutation(memory_size)
        else:
            rand_inds = np.random.randint(
                memory_size, size=size)
        exp_batch = [self.get_exp(i) for i in rand_inds]
        minibatch = self.format_minibatch(exp_batch)
        return minibatch
