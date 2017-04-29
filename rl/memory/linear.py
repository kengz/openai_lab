import numpy as np
from rl.memory.base_memory import Memory
from rl.util import log_self
from scipy.stats import halfnorm


class LinearMemory(Memory):

    '''
    The replay memory used for random minibatch training
    '''

    # absorb generic param without breaking
    def __init__(self, env_spec, **kwargs):
        super(LinearMemory, self).__init__(env_spec)
        self.exp_keys = [
            'states', 'actions', 'rewards', 'next_states', 'terminals']
        self.exp = {k: [] for k in self.exp_keys}
        log_self(self)

    def encode_action(self, action):
        '''encode action based on continuous/discrete before adding'''
        if self.agent.env_spec['actions'] == 'continuous':
            # continuous problem, keep as is
            encoded_action = action
        else:  # discrete problem
            if np.shape(action) == (self.env_spec['action_dim'], ):
                # raw action from continuous agent, keep as is
                # encoded_action = action
                # one-hot encode it
                encoded_action = np.zeros(self.agent.env_spec['action_dim'])
                encoded_action[np.argmax(action)] = 1
            else:  # action from discrete agent, do one-hot encoding
                encoded_action = np.zeros(self.agent.env_spec['action_dim'])
                encoded_action[action] = 1
        return encoded_action

    def add_exp(self, action, reward, next_state, terminal):
        '''
        after the env.step(a) that returns s', r,
        using the previously stored state for the s,
        form an experience tuple <s, a, r, s'>
        '''
        self.exp['states'].append(self.state)
        self.exp['actions'].append(self.encode_action(action))
        self.exp['rewards'].append(reward)
        self.exp['next_states'].append(next_state)
        self.exp['terminals'].append(int(terminal))
        self.state = next_state

    def _get_exp(self, exp_name, inds):
        return np.array([self.exp[exp_name][i] for i in inds])

    def get_exp(self, inds):
        return {k: self._get_exp(k, inds) for k in self.exp_keys}

    def pop(self):
        '''convenient method to get exp at [last_ind]'''
        assert self.size() > 0, 'memory is empty, cannot pop'
        return self.get_exp([self.size() - 1])

    def size(self):
        return len(self.exp['rewards'])

    def rand_minibatch(self, size):
        '''plain random sampling'''
        memory_size = self.size()
        rand_inds = np.random.randint(memory_size, size=size)
        minibatch = self.get_exp(rand_inds)
        return minibatch

    def update(self, updates):
        pass


class LinearMemoryWithForgetting(LinearMemory):

    '''
    Linear memory with uniform sampling, retaining last 50k experiences
    '''

    def __init__(self, env_spec, max_mem_len=50000,
                 **kwargs):  # absorb generic param without breaking
        super(LinearMemoryWithForgetting, self).__init__(env_spec)
        self.max_mem_len = max_mem_len

    def trim_exp(self):
        '''The forgetting mechanism'''
        if (self.size() > self.max_mem_len):
            for k in self.exp_keys:
                del self.exp[k][0]

    def add_exp(self, action, reward, next_state, terminal):
        '''
        add exp as usual, but preserve only the recent episodes
        '''
        super(LinearMemoryWithForgetting, self).add_exp(
            action, reward, next_state, terminal)
        self.trim_exp()


class LeftTailMemory(LinearMemory):

    '''
    Memory with sampling via a left-tail distribution
    '''

    def rand_minibatch(self, size):
        '''
        get a minibatch of random exp for training
        use simple memory decay, i.e. sample with a left tail
        distribution to draw more from latest memory
        then append with the most recent, untrained experience
        '''
        memory_size = self.size()
        new_exp_size = self.agent.train_per_n_new_exp
        if memory_size <= size or memory_size <= new_exp_size:
            inds = np.random.randint(memory_size, size=size)
        else:
            new_memory_ind = max(0, memory_size - new_exp_size)
            old_memory_ind = max(0, new_memory_ind - 1)
            latest_inds = np.arange(new_memory_ind, memory_size)
            random_batch_size = size - new_exp_size
            rand_inds = (old_memory_ind - halfnorm.rvs(
                size=random_batch_size,
                scale=float(old_memory_ind)*0.80).astype(int))
            inds = np.concatenate([rand_inds, latest_inds]).clip(0)
        minibatch = self.get_exp(inds)
        return minibatch
