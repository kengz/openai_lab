import numpy as np


class DQN(object):

    '''
    The simplest Q learner - a table,
    with epsilon-greedy method and
    Bellman equation for value.
    '''

    def __init__(self, env_spec,
                 resolution=10,
                 gamma=0.95,
                 init_e=1.0, final_e=0.1, e_anneal_steps=1000):
        self.env_spec = env_spec
        self.resolution = resolution
        self.gamma = gamma
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.e_anneal_steps = e_anneal_steps
        self.build_table()

    def build_table(self):
        '''
        init the 2D qtable by
        bijecting the state space into pixelated, flattened vector
        multiplied with
        list of possible discrete actions
        '''
        self.pixelate_state_space(self.resolution)
        flat_state_size = self.resolution ** self.env_spec['state_dim']
        self.qtable = np.random.uniform(
            low=-1, high=1, size=(flat_state_size, num_actions))
        return self.qtable

    def pixelate_state_space(self, resolution=10):
        '''chunk up the state space hypercube to specified resolution'''
        state_bounds = self.env_spec['state_bounds']
        self.state_pixels = [np.linspace(*sb, num=resolution+1)
                             for sb in state_bounds]
        return self.state_pixels

    def flatten_state(self, state):
        '''
        collapse a hyperdim state by binning into state_pixels
        then flattening the pixel_state into 1-dim bijection
        '''
        val_space_pairs = list(zip(state, self.state_pixels))
        pixel_state = [np.digitize(*val_space)
                       for val_space in val_space_pairs]  # binning
        flat_state = int("".join([str(ps) for ps in pixel_state]))
        return flat_state

    def train(self, replay_memory):
        '''
        step 1,2,3,4 of algo.
        replay_memory is provided externally
        '''
        # s,a,r,s'
        last_exp = replay_memory.get_exp(replay_memory.size() - 1)
        state = last_exp['states'][0]
        flat_state = self.flatten_state(state)
        action = last_exp['actions'][0]
        reward = last_exp['rewards'][0]
        self.qtable[flat_state, action] = (1 - self.alpha) * \
            self.qtable[flat_state, action] + \
            self.alpha * (reward + self.gamma *
                          self.qtable[flat_state, action])
        return self.qtable

    def update_e(self):
        '''strategy to update epsilon'''
        self.e = max(self.e -
                     (self.init_e - self.final_e)/float(self.e_anneal_steps),
                     self.final_e)
        return self.e

    def select_action(self, state):
        '''epsilon-greedy method'''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
        else:
            flat_state = self.flatten_state(state)
            action = self.qtable[flat_state].argsort()[-1]
        self.update_e()
        return action
