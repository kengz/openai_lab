import numpy as np
from rl.util import log_self
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

    def add_exp_processed(self, state, action, reward, next_state, terminal):
        '''add a processed experience'''
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
        log_self(self)

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

    def add_exp_processed(self, processed_state, action, reward,
                          processed_next_state, next_state, terminal):
        '''
        similar to add_exp function but allows for preprocessing raw state input
        E.g. concatenating states, diffing states, cropping, grayscale and stacking images
        '''
        self.exp['states'].append(processed_state)
        self.exp['actions'].append(self.one_hot_action(action))
        self.exp['rewards'].append(reward)
        self.exp['next_states'].append(processed_next_state)
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
    Linear memory with uniform sampling, retaining last 50k experiences
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

    def add_exp_processed(self, processed_state, action, reward,
                          processed_next_state, next_state, terminal):
        '''
        add processed exp as usual, but preserve only the recent episodes
        '''
        super(LinearMemoryWithForgetting, self).add_exp_processed(
            processed_state, action, reward,
            processed_next_state, next_state, terminal)

        if (self.size() > 50000):
            for k in self.exp_keys:
                del self.exp[k][0]


class LongLinearMemoryWithForgetting(LinearMemory):

    '''
    Linear memory with uniform sampling, retaining last 500k experiences
    '''

    def add_exp(self, action, reward, next_state, terminal):
        '''
        add exp as usual, but preserve only the recent episodes
        '''
        super(LongLinearMemoryWithForgetting, self).add_exp(
            action, reward, next_state, terminal)

        if (self.size() > 500000):
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


# if batch size too small wrt max timestep, it might not hit every position.
# rand minibatch needs to include those latest, unused training data

class RankedMemory(LinearMemory):

    '''
    Memory with ranking based on good or bad episodes
    experiences are grouped episodically
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        super(RankedMemory, self).__init__()
        # use the old self.exp as buffer, remember to clear
        self.last_exp = self.exp
        self.epi_memory = []
        self.sorted_epi_exp = self.exp
        self.n_best_epi = 10
        # then do left tail selection or early forget, I dont care
        log_self(self)

    # merge the epi_memory into an exp object
    def merge_exp(self):
        sorted_exp = {}
        # split epi_memory into better and worse halves
        half_epi_len = int(float(len(self.epi_memory))/float(2))
        for k in self.exp_keys:
            k_exp = np.concatenate(
                [epi_exp['exp'][k]
                    for epi_exp in self.epi_memory[-half_epi_len:]]
            )
            sorted_exp[k] = k_exp
        return sorted_exp

    def add_exp(self, action, reward, next_state, terminal):
        super(RankedMemory, self).add_exp(
            action, reward, next_state, terminal)
        if terminal:
            epi_exp = {
                'exp': self.exp,
                'total_rewards': np.sum(self.exp['rewards'])
            }
            self.epi_memory.append(epi_exp)
            self.epi_memory.sort(key=lambda epi_exp: epi_exp['total_rewards'])
            self.last_exp = self.exp
            self.exp = {k: [] for k in self.exp_keys}
            self.sorted_epi_exp = self.merge_exp()

    def pop(self):
        '''
        convenient method to get exp at [last_ind]
        '''
        buffer_exp = self.exp  # store for restore later
        self.exp = self.last_exp
        res = super(RankedMemory, self).pop()
        self.exp = buffer_exp
        return res

    def rand_minibatch(self, size):
        if len(self.epi_memory) == 0:   # base case, early exit
            return super(RankedMemory, self).rand_minibatch(size)

        buffer_exp = self.exp  # store for restoration after
        self.exp = self.sorted_epi_exp
        minibatch = super(RankedMemory, self).rand_minibatch(size)
        self.exp = buffer_exp  # set buffer back to original
        return minibatch

    def split_rand_minibatch(self, size):
        '''
        the minibatch composed of minibatches from the best epis
        guarantee that every exp will be trained at least once
        so always source the latest from buffer
        and then the rest from
        self.n_best_epi best epi_exp in epi_memory
        pick from buffer the new thing,
        store buffer, swap, pick for self.n_best_epi of them
        merge the minibatch
        set buffer back to original
        return minibatch
        '''
        new_exp_size = self.agent.train_per_n_new_exp
        if len(self.epi_memory) == 0:   # base case, early exit
            return super(RankedMemory, self).rand_minibatch(size)

        epi_memory_size = len(self.epi_memory)
        n_epi_exp = min(self.n_best_epi, epi_memory_size)
        epi_memory_start_ind = epi_memory_size - n_epi_exp
        # minibatch size to pick from an epi_exp
        epi_minibatch_size = max(1, np.int(np.ceil(size/n_epi_exp)))
        buffer_exp = self.exp  # store for restoration after

        best_epi_memory = []  # all the minibatches from the best epis
        # set self.exp to last n_best, pick epi_minibatch
        for i in range(epi_memory_start_ind, epi_memory_size):
            epi_exp = self.epi_memory[i]['exp']
            self.exp = epi_exp
            epi_minibatch = super(RankedMemory, self).rand_minibatch(
                epi_minibatch_size)
            best_epi_memory.append(epi_minibatch)

        self.exp = buffer_exp  # set buffer back to original
        if not self.pop()['terminals'][0]:
            new_minibatch = super(
                RankedMemory, self).rand_minibatch(new_exp_size)
            best_epi_memory.append(new_minibatch)

        # merge all minibatches from best_epi_memory into a minibatch
        minibatch = {}
        for k in self.exp_keys:
            k_exp = np.concatenate(
                [epi_exp[k] for epi_exp in best_epi_memory]
            )[-size:]
            minibatch[k] = k_exp
        assert len(minibatch['rewards']) == size

        return minibatch
