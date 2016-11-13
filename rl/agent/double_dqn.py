import numpy as np
from agent.dqn import DQN
from util import logger, pp
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


class DoubleDQN(DQN):

    '''
    The base class of double DQNs
    '''

    def build_net(self):
        logger.info(pp.pformat(self.env_spec))
        model = Sequential()
        model.add(Dense(4,
                        input_shape=(self.env_spec['state_dim'],),
                        init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        logger.info("Model 1 summary")
        model.summary()
        self.model = model

        model2 = Sequential()
        model2.add(Dense(4,
                         input_shape=(self.env_spec['state_dim'],),
                         init='lecun_uniform', activation='sigmoid'))
        model2.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        logger.info("Model 2 summary")
        model2.summary()
        self.model2 = model2
        return model, model2

    def build_graph(self):
        self.build_net()
        self.optimizer = SGD(lr=self.learning_rate)
        self.model.compile(
            loss='mean_squared_error', optimizer=self.optimizer)
        self.model2.compile(
            loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Models built and compiled")
        return self.model, self.model2

    def train(self, sys_vars, replay_memory):
        '''
        step 1,2,3,4 of algo.
        replay_memory is provided externally
        '''
        self.update_n_epoch(sys_vars)
        self.update_e(sys_vars, replay_memory)

        loss_total = 0
        for epoch in range(self.n_epoch):
            minibatch = replay_memory.rand_minibatch(self.batch_size)
            # algo step 1
            Q_states = self.model.predict(minibatch['states'])

            # Select max using model 2
            Q_next_states_select = self.model2.predict(
                minibatch['next_states'])
            Q_next_states_max_ind = np.argmax(Q_next_states_select, axis=1)
            # if more than one max, pick 1st
            if (Q_next_states_max_ind.shape[0] > 1):
                Q_next_states_max_ind = Q_next_states_max_ind[0]
            # Evaluate max using model 1

            # algo step 2
            Q_next_states = self.model.predict(minibatch['next_states'])
            Q_next_states_max = Q_next_states[:, Q_next_states_max_ind]

            # Q targets for batch-actions a;
            # with terminal to make future reward 0 if end
            Q_targets_a = minibatch['rewards'] + self.gamma * \
                (1 - minibatch['terminals']) * Q_next_states_max
            # set Q_targets of a as above
            # and the non-action units' Q_targets to as-is
            # minibatch['actions'] is one-hot encoded
            Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
                (1 - minibatch['actions']) * Q_states

            # logger.info("minibatch actions: {}\n Q_targets_a (reshapes): {}"
            #             "\n Q_states: {}\n Q_targets: {}\n\n".format(
            #                 minibatch['actions'], Q_targets_a[
            #                     :, np.newaxis], Q_states,
            #                 Q_targets))

            loss = self.model.train_on_batch(minibatch['states'], Q_targets)
            loss_total += loss

            # Switch model 1 and model 2
            temp = self.model
            self.model = self.model2
            self.model2 = temp

        return loss_total / self.n_epoch
