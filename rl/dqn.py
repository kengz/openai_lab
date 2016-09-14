import math
import tflearn
import tensorflow as tf
import numpy as np
from copy import deepcopy


class DQN(object):

    '''
    The simplest deep Q network. See DQN.md
    '''

    def __init__(self, env_spec, session,
                 gamma=0.95, learning_rate=0.1,
                 init_e=1.0, final_e=0.1, e_anneal_steps=100,
                 batch_size=64):
        self.env_spec = env_spec
        self.session = session
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.e_half_life = e_anneal_steps
        self.batch_size = batch_size
        self.build_graph()

    def build_net(self):
        # X = tflearn.input_data(shape=[None, self.env_spec['state_dim']])
        # reshape into 3D tensor for conv
        # net = tf.reshape(X, [-1, self.env_spec['state_dim'], 1])
        # net = tflearn.conv_1d(net, 8, 2, activation='relu')
        # net = tflearn.fully_connected(net, 16, activation='relu')
        # net = tflearn.dropout(net, 0.5)
        # no conv
        X = tf.placeholder(
            "float", [None, self.env_spec['state_dim']])
        net = tflearn.fully_connected(X, 8, activation='relu')
        # net = tflearn.dropout(net, 0.5)
        # net = tflearn.fully_connected(net, 8, activation='relu')
        # net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(
            net, self.env_spec['action_dim'])
        self.X = X
        self.net = net
        return net

    def build_graph(self):
        net = self.build_net()
        self.Q_target = tf.placeholder(
            "float", [None, self.env_spec['action_dim']])
        # target Y - predicted Y, do rms loss
        self.loss = tf.reduce_mean(tf.square(self.Q_target - self.net))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        return self.train_op

    def train(self, replay_memory):
        '''
        step 1,2,3,4 of algo. plural for batch's multiple values
        replay_memory is provided externally
        '''
        self.update_e(replay_memory)
        minibatch = replay_memory.rand_minibatch(self.batch_size)
        # algo step 1
        Q_states = self.net.eval(
            feed_dict={self.X: minibatch['states']}, session=self.session)
        # algo step 2
        Q_next_states = self.net.eval(
            feed_dict={self.X: minibatch['next_states']}, session=self.session)
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

    def update_e(self, replay_memory):
        '''
        strategy to update epsilon
        '''
        global_time_step = replay_memory.size()
        unscaled_e = self.init_e * \
            math.exp(-.693/self.e_half_life*float(global_time_step))
        # rescale to fit in 0.1 to 1.0, translated + 0.1
        self.e = unscaled_e*abs(self.init_e - self.final_e) + self.final_e
        return self.e

    def select_action(self, state):
        '''
        step 1 of algo, feedforward
        '''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
        else:
            Q_state = self.net.eval(
                feed_dict={self.X: [state]}, session=self.session)
            action = self.session.run(tf.argmax(Q_state, 1))[0]
        return action

    def save(self, model_path, global_step):
        self.saver = tf.train.Saver(tf.trainable_variables())
        # print(len(tf.trainable_variables()))
        # print(len(tf.all_variables()))
        # for v in tf.trainable_variables():
        # for v in tf.all_variables():
        #     print(v.name)
        #     print(v.value)
        #     print(dir(v))
        #     print(self.session.run(v))
        # !set global_step None to save model without -N
        return self.saver.save(
            self.session, model_path, global_step=global_step)
        # proxy used for saving model
        # trainop = tflearn.TrainOp(loss=self.loss, optimizer=self.optimizer)
        # trainer = tflearn.Trainer(
        # train_ops=trainop, tensorboard_verbose=3, session=self.session)
    # return trainer.save(model_path, global_step=len(replay_memory.memory))

    def restore(self, model_path):
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.saver.restore(self.session, model_path)
