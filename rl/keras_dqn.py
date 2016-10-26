import numpy as np
np.random.seed(42)

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l1, l2
from keras.constraints import maxnorm
from keras.objectives import categorical_crossentropy
from keras.optimizers import SGD


class DQN(object):

    '''
    The simplest deep Q network. See DQN.md
    '''

    def __init__(self, env_spec, session,
                 gamma=0.95, learning_rate=0.1,
                 init_e=1.0, final_e=0.1, e_anneal_steps=1000,
                 batch_size=64, n_epoch=2):
        self.env_spec = env_spec
        self.session = session
        K.set_session(session)
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
        X = tf.placeholder(tf.float32, shape=(None, self.env_spec['state_dim']))
        model = Sequential()
        model.add(Dense(4, input_shape=(self.env_spec['state_dim'],), init='lecun_uniform'))
        model.add(Dense(2, init='lecun_uniform'))
        # model.add(Dropout(0.5)) # will break wtf
        # model.add(Dense(128, activation='relu', init='lecun_uniform', W_regularizer=l1(0.01)))
        # model.add(Dense(64, activation='relu', init='lecun_uniform', W_regularizer=l1(0.01)))
        # model.add(Dropout(0.5))
        # model.add(Dense(32, activation='relu', init='lecun_uniform', W_regularizer=l1(0.01)))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        model.summary()
        net = model(X)
        self.X = X
        self.model = model
        self.net = net
        return net

    def build_graph(self):
        net = self.build_net()
        self.Q_target = tf.placeholder(tf.float32, shape=(None, self.env_spec['action_dim']))
        # target Y - predicted Y, do rms loss
        self.loss = tf.reduce_mean(categorical_crossentropy(self.Q_target, self.net))
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.initialize_all_variables())
        return self.train_op

    def train(self, replay_memory):
        '''
        step 1,2,3,4 of algo. plural for batch's multiple values
        replay_memory is provided externally
        '''
        minibatch = replay_memory.rand_minibatch(self.batch_size)
        for epoch in range(self.n_epoch):
            # algo step 1
            Q_states = self.net.eval(
                feed_dict={self.X: minibatch['states']},
                session=self.session)
            # algo step 2
            Q_next_states = self.net.eval(
                feed_dict={self.X: minibatch['next_states']},
                session=self.session)
            Q_next_states_max = np.amax(Q_next_states, axis=1)
            # Q targets for batch-actions a;
            # with terminal to make future reward 0 if end
            Q_targets_a = minibatch['rewards'] + self.gamma * \
                (1 - minibatch['terminals']) * Q_next_states_max
            # set Q_targets of a as above, and the non-action units' Q_targets to
            # as from algo step 1
            Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
                (1 - minibatch['actions']) * Q_states

            _, loss = self.session.run([self.train_op, self.loss], feed_dict={
                self.X: minibatch['states'],
                self.Q_target: Q_targets,
            })
        return loss

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
            print('random act')
        else:
            Q_state = self.net.eval(
                feed_dict={self.X: [state]}, session=self.session)
            action = self.session.run(tf.argmax(Q_state, 1))[0]
            print('---')
        self.update_e()
        return action

    def save(self, model_path, global_step=None):
        print('Saving model checkpoint')
        self.model.save_weights(model_path)

    def restore(self, model_path):
        self.model.load_weights(model_path, by_name=False)
