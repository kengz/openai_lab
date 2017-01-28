import numpy as np
from rl.agent.dqn import DQN


class DeepExpectedSarsa(DQN):

    '''
    Deep Expected Sarsa agent.
    On policy, with updates after each experience
    Policy = epsilonGreedyPolicy
    '''
    def __init__(self, *args, **kwargs):
        super(DeepExpectedSarsa, self).__init__(*args, **kwargs)
        self.train_per_n_new_exp = 1
        self.batch_size = 1
        self.n_epoch = 1
        self.final_n_epoch = 1

    def compute_Q_states(self, last_exp):
        clip_val = 10000
        Q_states = np.clip(
            self.model.predict(last_exp['states']), -clip_val, clip_val)
        Q_next_states = np.clip(
            self.model.predict(last_exp['next_states']), -clip_val, clip_val)
        
        curr_e = self.policy.e
        curr_e_per_a = curr_e / self.env_spec['action_dim']

        Q_next_states_max = np.amax(Q_next_states, axis=1)
        expected_Q = (1 - curr_e) * Q_next_states_max + \
                        np.sum(Q_next_states * curr_e_per_a, axis=1)
        return (Q_states, expected_Q)

    def train_an_epoch(self):
        last_exp = self.memory.pop()
        (Q_states, expected_Q) = self.compute_Q_states(last_exp)
        Q_targets = self.compute_Q_targets(
            last_exp, Q_states, expected_Q)
        loss = self.model.train_on_batch(last_exp['states'], Q_targets)
        return loss