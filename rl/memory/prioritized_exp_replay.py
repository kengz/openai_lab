import numpy as np
from rl.memory.linear import LinearMemoryWithForgetting
from rl.util import log_self

class PrioritizedExperienceReplay(LinearMemoryWithForgetting):
    '''
    Replay memory with random sampling weighted by the absolute
    size of the value function error
    
    Adapted from https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    memory unit
    '''
    def __init__(self, e, alpha, mem_max_len,
                            **kwargs): 
        super(PrioritizedExperienceReplay, self).__init__()
        # Prevents experiences with error of 0 from being replayed
        self.e = e
        # Controls how spiked the distribution is. alpha = 0 corresponds to uniform
        self.alpha = alpha 
        self.curr_data_inds = None
        self.curr_tree_inds = None
        self.mem_max_len = mem_max_len
        self.prio_tree = SumTree(self.mem_max_len)

    def get_priority(self, error):
        return (error + self.e) ** self.alpha

    def add_exp(self, action, reward, next_state, terminal, error):
        super(PrioritizedExperienceReplay, self).add_exp(
            action, reward, next_state, terminal, error)
        
        p = self.get_priority(error)
        self.prio_tree.add(p)

        # TO DO: How to update both consistently when 
        # memoy units are at capacity

    def rand_minibatch(self, size):
        '''
        plain random sampling, weighted by priority
        '''
        memory_size = self.size()
        self.curr_tree_inds, self.curr_data_inds = self.select_prio_inds(size)

        # print("CURRENT MEM INDS")
        # print(self.curr_data_inds)

        minibatch = self.get_exp(self.curr_data_inds)
        return minibatch

    def select_prio_inds(self, size):
        tree_inds = []
        data_inds = []

        # print("TREE TOTAL")
        # print(self.prio_tree.total())
        # print("TREE")
        # print(self.prio_tree.tree)

        segment = self.prio_tree.total() / size

        for i in range(size):
            # print("SEGMENT")
            # print(segment)

            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            t_idx, d_idx = self.prio_tree.get(s)
            tree_inds.append(t_idx)
            data_inds.append(d_idx)

        return tree_inds, data_inds

    def update(self, updates):
        # print("UPDATES")
        # print(updates)
        for i in range(len(updates)):
            t_idx = self.curr_tree_inds[i]
            d_idx = self.curr_data_inds[i]
            p = self.get_priority(updates[i])
            # print("Updating index {} with {}".format(d_idx, p))
            self.prio_tree.update(t_idx, p)
            self.exp['error'][d_idx] = updates[i]


class SumTree(object):
    '''
    Adapted from  https://github.com/jaara/AI-blog/blob/master/SumTree.py
    See https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    for a good introduction to PER
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )

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
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        # print("IDX")
        # print(idx)
        # print(self.tree[idx])
        # print("Dataidx")
        # print(data_idx)

        return idx, data_idx