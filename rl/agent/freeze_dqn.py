import numpy as np
from os import getpid
from rl.agent.double_dqn import DoubleDQN
from rl.agent.dqn import DQN
from rl.util import logger, clone_model


class FreezeDQN(DoubleDQN):

    '''
    Extends DQN agent to freeze target Q network
    and periodically update them to the weights of the
    exploration model
    Avoids oscillations and breaks correlation
    between Q-network and target
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf
    Exploration model periodically cloned into target Q network
    '''

    def compute_Q_states(self, minibatch):
        Q_states = np.clip(self.model.predict(minibatch['states']),
                           -self.clip_val, self.clip_val)
        Q_next_states = np.clip(self.model_2.predict(minibatch['next_states']),
                                -self.clip_val, self.clip_val)
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        return (Q_states, Q_next_states, Q_next_states_max)

    def train_an_epoch(self):
        # Should call DQN to train an epoch, not DoubleDQN
        return DQN.train_an_epoch(self)

    def update_target_model(self):
        # Also, loading logic seems off
        self.model_2 = clone_model(self.model)
        logger.debug("Updated target model weights")

    def update(self, sys_vars):
        '''
        Agent update apart from training the Q function
        '''
        done = sys_vars['done']
        timestep_check = sys_vars['t'] == (self.env_spec['timestep_limit'] - 1)
        if done or timestep_check:
            self.update_target_model()
        super(FreezeDQN, self).update(sys_vars)
