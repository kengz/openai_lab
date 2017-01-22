import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger
from keras.models import Sequential


class DoubleDQN(DQN):

    '''
    The base class of double DQNs
    '''

    def build_model(self):
        super(DoubleDQN, self).build_model()

        model2 = Sequential.from_config(self.model.get_config())
        logger.info("Model 2 summary")
        model2.summary()
        self.model2 = model2

        self.model2.compile(
            loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Models built and compiled")
        return self.model, self.model2

    def compute_Q_states(self, minibatch):
        Q_states = self.model.predict(minibatch['states'])
        # Different from (single) dqn: Select max using model 2
        Q_next_states_select = self.model2.predict(
            minibatch['next_states'])
        Q_next_states_max_ind = np.argmax(Q_next_states_select, axis=1)
        # if more than one max, pick 1st
        if (Q_next_states_max_ind.shape[0] > 1):
            Q_next_states_max_ind = Q_next_states_max_ind[0]

        # same as dqn again, but use Q_next_states_max_ind above
        Q_next_states = self.model.predict(minibatch['next_states'])
        Q_next_states_max = Q_next_states[:, Q_next_states_max_ind]
        return (Q_states, Q_next_states_max)

    def train(self, sys_vars):
        loss_total = 0
        for _epoch in range(self.n_epoch):
            # same as dqn
            minibatch = self.memory.rand_minibatch(self.batch_size)

            (Q_states, Q_next_states_max) = self.compute_Q_states(minibatch)
            Q_targets = self.compute_Q_targets(
                minibatch, Q_states, Q_next_states_max)

            loss = self.model.train_on_batch(minibatch['states'], Q_targets)
            loss_total += loss

            # Switch model 1 and model 2
            temp = self.model
            self.model = self.model2
            self.model2 = temp
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss
