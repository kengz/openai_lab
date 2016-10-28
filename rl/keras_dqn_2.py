import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l1, l2
from keras.constraints import maxnorm
from keras.objectives import mse


class DQN(object):

    '''
    The simplest deep Q network. See DQN.md
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
        model = Sequential()
        # Not clear how much better the algorithm is with regularization
        model.add(Dense(self.env_spec['state_dim'],
                        input_shape=(self.env_spec['state_dim'],),
                        init='lecun_uniform', activation='sigmoid'))
        # model.add(Dense(2, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        model.summary()
        self.model = model
        return model

    def build_graph(self):
        self.build_net()
        self.optimizer = SGD(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        print("Model built and compiled")
        return self.model

    def train(self, replay_memory):
        '''
        step 1,2,3,4 of algo. plural for batch's multiple values
        replay_memory is provided externally
        '''
        loss_total = 0
        minibatch = replay_memory.rand_minibatch(self.batch_size)
        for epoch in range(self.n_epoch):
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
            # set Q_targets of a as above,
            # and the non-action units' Q_targets to
            # as from algo step 1
            Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
                (1 - minibatch['actions']) * Q_states

            # print("minibatch actions: {}\n Q_targets_a (reshapes): {}\n Q_states: {}\n Q_targets: {}\n\n".format(
            #                minibatch['actions'], Q_targets_a[:, np.newaxis], Q_states, Q_targets))

            loss = self.model.train_on_batch(minibatch['states'], Q_targets)
            loss_total += loss
        return loss_total

    def update_e(self):
        '''
        strategy to update epsilon
        '''
        self.e = max(self.e -
                     (self.init_e - self.final_e)/float(self.e_anneal_steps),
                     self.final_e)
        return self.e

    def select_action(self, state):
        '''
        step 1 of algo, feedforward
        '''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
            # hmm maybe flip by ep?
            # print('random act')
        else:
            #print("state shape: {}".format(state.shape))
            state = np.reshape(state, (1, state.shape[0]))
            Q_state = self.model.predict(state)
            action = np.argmax(Q_state)
            # print('---')
        self.update_e()
        return action

    def save(self, model_path, global_step=None):
        print('Saving model checkpoint')
        self.model.save_weights(model_path)

    def restore(self, model_path):
        self.model.load_weights(model_path, by_name=False)
