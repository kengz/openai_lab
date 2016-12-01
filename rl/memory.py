import numpy as np
from rl.util import logger, pp
from scipy.stats import halfnorm


class Memory(object):

    '''
    The base class of Memory, with the core methods
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        '''
        Construct externally, and set at Agent.compile()
        '''
        self.agent = None
        self.state = None

    def reset_state(self, init_state):
        '''
        reset the state of LinearMemory per episode env.reset()
        '''
        self.state = init_state

    def add_exp(self, action, reward, next_state, terminal):
        '''add an experience'''
        raise NotImplementedError()

    def get_exp(self, inds):
        '''get a batch of experiences by indices'''
        raise NotImplementedError()

    def pop(self):
        '''get the last experience (batched like get_exp()'''
        raise NotImplementedError()

    def size(self):
        '''get a batch of experiences by indices'''
        raise NotImplementedError()

    def rand_minibatch(self, size):
        '''get a batch of experiences by indices'''
        raise NotImplementedError()


class LinearMemory(Memory):

    '''
    The replay memory used for random minibatch training
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        super(LinearMemory, self).__init__()
        self.exp_keys = [
            'states', 'actions', 'rewards', 'next_states', 'terminals']
        self.exp = {k: [] for k in self.exp_keys}
        logger.info('Memory params: {}'.format(pp.pformat(self.__dict__)))

    def one_hot_action(self, action):
        action_arr = np.zeros(self.agent.env_spec['action_dim'])
        action_arr[action] = 1
        return action_arr

    def add_exp(self, action, reward, next_state, terminal):
        '''
        after the env.step(a) that returns s', r,
        using the previously stored state for the s,
        form an experience tuple <s, a, r, s'>
        '''
        self.exp['states'].append(self.state)
        self.exp['actions'].append(self.one_hot_action(action))
        self.exp['rewards'].append(reward)
        self.exp['next_states'].append(next_state)
        self.exp['terminals'].append(int(terminal))
        self.state = next_state

    def _get_exp(self, exp_name, inds):
        return np.array([self.exp[exp_name][i] for i in inds])

    def get_exp(self, inds):
        return {k: self._get_exp(k, inds) for k in self.exp_keys}

    def pop(self):
        '''
        convenient method to get exp at [last_ind]
        '''
        assert self.size() > 0
        return self.get_exp([self.size() - 1])

    def size(self):
        return len(self.exp['rewards'])

    def rand_minibatch(self, size):
        '''
        plain random sampling
        '''
        memory_size = self.size()
        rand_inds = np.random.randint(memory_size, size=size)
        minibatch = self.get_exp(rand_inds)
        return minibatch


class LinearMemoryWithForgetting(LinearMemory):

    '''
    Linear memory with uniform sampling, retaining last 20k experiences
    '''

    def add_exp(self, action, reward, next_state, terminal):
        '''
        add exp as usual, but preserve only the recent episodes
        '''
        super(LinearMemoryWithForgetting, self).add_exp(
            action, reward, next_state, terminal)

        if (self.size() > 50000):
            for k in self.exp_keys:
                del self.exp[k][0]


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
        # increase to k if we skip training to every k time steps
        latest_batch_size = 1
        new_memory_ind = max(0, memory_size - latest_batch_size)
        old_memory_ind = max(0, new_memory_ind - 1)
        latest_inds = np.arange(new_memory_ind, memory_size)
        random_batch_size = size - latest_batch_size
        rand_inds = (old_memory_ind - halfnorm.rvs(
            size=random_batch_size,
            scale=float(old_memory_ind)*0.37).astype(int))
        inds = np.concatenate([latest_inds, rand_inds]).clip(0)
        minibatch = self.get_exp(inds)
        return minibatch
