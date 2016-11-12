import numpy as np
from util import logger, pp
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
# from keras.regularizers import l1, l2
# from keras.constraints import maxnorm
# from keras.objectives import mse


class DQN(object):

    '''
    The simplest deep Q network,
    with epsilon-greedy method and
    Bellman equation for value, using neural net.
    '''

    def __init__(self, env_spec,
                 gamma=0.95, learning_rate=0.1,
                 init_e=1.0, final_e=0.1, e_anneal_steps=1000,
                 batch_size=16, n_epoch=1):
        self.env_spec = env_spec
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.e_anneal_steps = e_anneal_steps
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.build_graph()

    def build_net(self):
        logger.info(pp.pformat(self.env_spec))
        model = Sequential()
        # Not clear how much better the algorithm is with regularization
        model.add(Dense(8,
                        input_shape=(self.env_spec['state_dim'],),
                        init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(6, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        model.summary()
        self.model = model
        return model

    def build_graph(self):
        self.build_net()
        self.optimizer = SGD(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Model built and compiled")
        return self.model

    def select_action(self, state):
        '''epsilon-greedy method'''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
        else:
            state = np.reshape(state, (1, state.shape[0]))
            Q_state = self.model.predict(state)
            action = np.argmax(Q_state)
        return action

    def update_e(self, sys_vars, replay_memory):
        '''strategy to update epsilon'''
        epi = sys_vars['epi']
        mem_size = replay_memory.size()
        rise = self.final_e - self.init_e
        slope = rise / float(self.e_anneal_steps)
        self.e = max(slope * mem_size + self.init_e, self.final_e)
        if not (epi % 2) and epi > 15:
            # drop to 1/3 of the current exploration rate
            self.e = max(self.e/3., self.final_e)
        return self.e

    def train(self, sys_vars, replay_memory):
        '''
        step 1,2,3,4 of algo.
        replay_memory is provided externally
        '''
        self.update_e(sys_vars, replay_memory)

        loss_total = 0
        for epoch in range(self.n_epoch):
            minibatch = replay_memory.rand_minibatch(self.batch_size)
            # algo step 1
            Q_states = self.model.predict(minibatch['states'])
            # algo step 2
            Q_next_states = self.model.predict(minibatch['next_states'])
            # batch x num_actions
            Q_next_states_max = np.amax(Q_next_states, axis=1)
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
        return loss_total / self.n_epoch

    def save(self, model_path, global_step=None):
        logger.info('Saving model checkpoint')
        self.model.save_weights(model_path)

    def restore(self, model_path):
        self.model.load_weights(model_path, by_name=False)
