import numpy as np
from rl.util import log_self
from scipy.stats import halfnorm
import math


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

class HighLowMemory(LinearMemory):

    '''
    Memory divided into two: good and bad experiences
    As with RankedMemory experiences are grouped episodically
    Episodes with a total reward > threshold are assigned to good memory
    The threshold is recomputed every n episodes and 
    episodes are reassigned accordingly.
    Memories are sampled from good experiences with a self.prob_high
    Memories are sampled from bad experiences with a 1 - self.prob_high
    Experiences are sampled from a maximum of 3 randomly selected episodes,
    per minibatch for each of the high and low memories
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        super(HighLowMemory, self).__init__()
        # use the old self.exp as buffer, remember to clear
        self.last_exp = self.exp
        self.epi_memory_high = []
        self.epi_memory_low = []
        self.max_reward =  -math.inf
        self.min_reward = math.inf
        self.threshold = math.inf # 1st  5 epis goes into bad half, recompute every 5 epis
        self.threshold_history = []
        self.epi_num = 0
        self.prob_high = 0.66
        self.num_epis_to_sample = 3
        self.max_epis_in_mem = 15
        self.recompute_freq = 10
        log_self(self)


    def reassign_episodes(self):
        temp_high = self.epi_memory_high
        temp_low = self.epi_memory_low
        self.epi_memory_high = []
        self.epi_memory_low = []

        for epi in temp_high:
            if (epi['total_rewards'] > self.threshold):
                self.epi_memory_high.append(epi)
            else:
                self.epi_memory_low.append(epi)

        for epi in temp_low:
            if (epi['total_rewards'] > self.threshold):
                self.epi_memory_high.append(epi)
            else:
                self.epi_memory_low.append(epi)

    def compute_threshold(self):
        self.threshold_history.append([self.threshold, 
                                                                self.max_reward,
                                                                self.min_reward])
        if (len(self.threshold_history) > 1):
        # Scaled because this threshold seems too severe based on trial runs
            self.threshold =  \
                max(self.threshold, (self.max_reward + self.min_reward) / 2.0 * 0.75)
        else:
            self.threshold =  (self.max_reward + self.min_reward) / 2.0 * 0.75
        self.reassign_episodes()
        self.max_reward =  -math.inf
        self.min_reward = math.inf


    def add_exp(self, action, reward, next_state, terminal):
        super(HighLowMemory, self).add_exp(
            action, reward, next_state, terminal)
        if terminal:
            epi_exp = {
                'exp': self.exp,
                'total_rewards': np.sum(self.exp['rewards']),
                'epi_num' : self.epi_num
            }
            if (epi_exp['total_rewards']  <= self.threshold):
                self.epi_memory_low.append(epi_exp)
            else:
                self.epi_memory_high.append(epi_exp)
            if (self.epi_num > 0 and self.epi_num % self.recompute_freq == 0):
                self.compute_threshold()
            if (epi_exp['total_rewards'] > self.max_reward):
                self.max_reward = epi_exp['total_rewards']
            if (epi_exp['total_rewards'] < self.min_reward):
                self.min_reward = epi_exp['total_rewards'] 
            self.last_exp = self.exp
            self.exp = {k: [] for k in self.exp_keys}
            self.epi_num += 1
            # print("THRESHOLD HISTORY")
            # print(self.threshold_history)
            # print("HIGH MEM")
            # for epi in self.epi_memory_high:
            #     print(str(epi['total_rewards'])+ " ,", end=" ")
            # print()
            # print("LOW MEM")
            # for epi in self.epi_memory_low:
            #     print(str(epi['total_rewards'] )+ " ,", end=" ")
            # print()

            
    def add_exp_processed(self, processed_state, action, reward, 
                                processed_next_state, next_state, terminal):
        super(HighLowMemory, self).add_exp_processed(
                    processed_state, action, reward, 
                    processed_next_state, next_state, terminal)
        if terminal:
            epi_exp = {
                'exp': self.exp,
                'total_rewards': np.sum(self.exp['rewards']),
                'epi_num' : self.epi_num
            }
            if (epi_exp['total_rewards']  <= self.threshold):
                self.epi_memory_low.append(epi_exp)
            else:
                self.epi_memory_high.append(epi_exp)
            if (self.epi_num > 0 and self.epi_num % self.recompute_freq == 0):
                self.compute_threshold()
            if (epi_exp['total_rewards'] > self.max_reward):
                self.max_reward = epi_exp['total_rewards']
            if (epi_exp['total_rewards'] < self.min_reward):
                self.min_reward = epi_exp['total_rewards'] 
            self.last_exp = self.exp
            self.exp = {k: [] for k in self.exp_keys}
            self.epi_num += 1
            # print("THRESHOLD HISTORY")
            # print(self.threshold_history)
            # print("HIGH MEM")
            # for epi in self.epi_memory_high:
            #     print(str(epi['total_rewards'])+ " ,", end=" ")
            # print()
            # print("LOW MEM")
            # for epi in self.epi_memory_low:
            #     print(str(epi['total_rewards'] )+ " ,", end=" ")
            # print()
            

    def pop(self):
        '''
        convenient method to get exp at [last_ind]
        '''
        buffer_exp = self.exp  # store for restore later
        self.exp = self.last_exp
        res = super(HighLowMemory, self).pop()
        self.exp = buffer_exp
        return res

    def rand_minibatch(self, size):
        # base case, early exit
        high_samples = np.int(np.ceil(size * self.prob_high))
        low_samples = size - high_samples

        if (len(self.epi_memory_high) == 0) and (len(self.epi_memory_low) == 0):   
            return super(HighLowMemory, self).rand_minibatch(size)

        if (len(self.epi_memory_high) == 0):
            high_samples = 0
            low_samples = size

        high_samples_per_epi = np.int(np.ceil(high_samples / self.num_epis_to_sample))
        low_samples_per_epi = np.int(np.ceil(low_samples / self.num_epis_to_sample))

        buffer_exp = self.exp
        minibatch_as_list = []
        if high_samples > 0:
            for i in range(4):
                idx = np.random.randint(0, len(self.epi_memory_high))
                epi_exp = self.epi_memory_high[idx]['exp']
                self.exp = epi_exp
                epi_minibatch = super(HighLowMemory, self).rand_minibatch(
                    high_samples_per_epi)
                minibatch_as_list.append(epi_minibatch)

        if low_samples > 0:
            for i in range(4):
                idx = np.random.randint(0, len(self.epi_memory_low))
                epi_exp = self.epi_memory_low[idx]['exp']
                self.exp = epi_exp
                epi_minibatch = super(HighLowMemory, self).rand_minibatch(
                    low_samples_per_epi)
                minibatch_as_list.append(epi_minibatch)

        # set buffer back to original
        self.exp = buffer_exp

        # merge all minibatches from best_epi_memory into a minibatch
        minibatch = {}
        for k in self.exp_keys:
            k_exp = np.concatenate(
                [epi_exp[k] for epi_exp in minibatch_as_list]
            )[-size:]
            minibatch[k] = k_exp
        assert len(minibatch['rewards']) == size

        return minibatch

class HighLowMemoryWithForgetting(HighLowMemory):

    '''
    Like HighLowMemory but also has forgetting capability
    Controlled by max_epis_in_mem param
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        super(HighLowMemoryWithForgetting, self).__init__()
        self.max_epis_in_mem = 250
        log_self(self)

    def reassign_episodes(self):
        temp_high = self.epi_memory_high
        temp_low = self.epi_memory_low
        self.epi_memory_high = []
        self.epi_memory_low = []

        for epi in temp_high:
            if (self.epi_num - epi['epi_num'] <= self.max_epis_in_mem):
                if (epi['total_rewards'] > self.threshold):
                    self.epi_memory_high.append(epi)
                else:
                    self.epi_memory_low.append(epi)

        for epi in temp_low:
             if (self.epi_num - epi['epi_num'] <= self.max_epis_in_mem):
                if (epi['total_rewards'] > self.threshold):
                    self.epi_memory_high.append(epi)
                else:
                    self.epi_memory_low.append(epi)

        




    
