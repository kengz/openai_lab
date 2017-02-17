import os
import numpy as np
from rl.agent.double_dqn import DoubleDQN
from rl.agent.dqn import DQN
from keras.models import load_model
from rl.util import logger


class DQNFreeze(DoubleDQN):

    '''
    Extends DQN agent to freeze target Q network
    and periodically update them to the weights of the
    exploration model
    Avoids oscillations and breaks correlation
    between Q-network and target
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf
    Exploration model periodically saved and loaded into target Q network
    '''

    def compute_Q_states(self, minibatch):
        Q_states = np.clip(self.model.predict(minibatch['states']),
                           -self.clip_val, self.clip_val)
        Q_next_states = np.clip(self.model2.predict(minibatch['next_states']),
                                -self.clip_val, self.clip_val)
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        return (Q_states, Q_next_states, Q_next_states_max)

    def train_an_epoch(self):
        # Should call DQN to train an epoch, not DoubleDQN
        return DQN.train_an_epoch(self)

    def update_target_model(self):
        pid = os.getpid()
        name = 'temp_Q_model_freeze_' + str(pid) + '.h5'
        self.model.save(name)
        self.model2 = load_model(name)
        logger.info("Updated target model weights")

    def update(self, sys_vars):
        '''
        Agent update apart from training the Q function
        '''
        done = sys_vars['done']
        timestep_check = sys_vars['t'] == (self.env_spec['timestep_limit'] - 1)
        if done or timestep_check:
            self.update_target_model()
        super(DQNFreeze, self).update(sys_vars)
