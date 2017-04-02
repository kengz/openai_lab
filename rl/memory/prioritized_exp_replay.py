import numpy as np
from rl.memory.linear import LinearMemoryWithForgetting


class PrioritizedExperienceReplay(LinearMemoryWithForgetting):

    '''
    Replay memory with random sampling weighted by the absolute
    size of the value function error

    Adapted from https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    memory unit
    '''

    def __init__(self, max_mem_len=10000, e=0.01, alpha=0.6,
                 **kwargs):
        super(PrioritizedExperienceReplay, self).__init__(max_mem_len)
        # Prevents experiences with error of 0 from being replayed
        self.e = e
        # Controls how spiked the distribution is. alpha = 0 means uniform
        self.alpha = alpha
        self.curr_data_inds = None
        self.curr_tree_inds = None
        self.prio_tree = SumTree(self.max_mem_len)
        self.head = 0

    def get_priority(self, error):
        return (error + self.e) ** self.alpha

    def add_exp(self, action, reward, next_state, terminal, error):
        '''Round robin memory updating'''
        if self.size() < self.max_mem_len:  # add as usual
            super(PrioritizedExperienceReplay, self).add_exp(
                action, reward, next_state, terminal, error)
        else:  # replace round robin
            self.exp['states'][self.head] = self.state
            self.exp['actions'][self.head] = self.encode_action(action)
            self.exp['rewards'][self.head] = reward
            self.exp['next_states'][self.head] = next_state
            self.exp['terminals'][self.head] = int(terminal)
            self.exp['error'][self.head] = error
            self.state = next_state

        self.head += 1
        if self.head >= self.max_mem_len:
            self.head = 0  # reset for round robin

        p = self.get_priority(error)
        self.prio_tree.add(p)

        assert self.head == self.prio_tree.head

    def rand_minibatch(self, size):
        '''
        plain random sampling, weighted by priority
        '''
        self.curr_tree_inds, self.curr_data_inds = self.select_prio_inds(size)
        minibatch = self.get_exp(self.curr_data_inds)
        return minibatch

    def select_prio_inds(self, size):
        tree_inds = []
        data_inds = []
        segment = self.prio_tree.total() / size

        for i in range(size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            t_idx, d_idx = self.prio_tree.get(s)
            tree_inds.append(t_idx)
            data_inds.append(d_idx)

        return tree_inds, data_inds

    def update(self, updates):
        for i, u in enumerate(updates):
            t_idx = self.curr_tree_inds[i]
            d_idx = self.curr_data_inds[i]
            p = self.get_priority(u)
            self.prio_tree.update(t_idx, p)
            self.exp['error'][d_idx] = u


class SumTree(object):

    '''
    Adapted from  https://github.com/jaara/AI-blog/blob/master/SumTree.py
    See https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    for a good introduction to PER
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.head = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p):
        idx = self.head + self.capacity - 1

        self.update(idx, p)

        self.head += 1
        if self.head >= self.capacity:
            self.head = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, data_idx
