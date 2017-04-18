""" 
DDPG implementation from https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger, clone_model, clone_optimizer

from rl.agent.base_agent import Agent
import tensorflow as tf
# import gym
# from gym import wrappers
import tflearn

# from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# # ===========================
# #   Utility Parameters
# # ===========================
# # Render gym env during training
# RENDER_ENV = True
# # Use Gym Monitor
# GYM_MONITOR_EN = True
# # Gym environment
# ENV_NAME = 'Pendulum-v0'
# # Directory for storing gym results
# MONITOR_DIR = './results/gym_ddpg'
# # Directory for storing tensorboard summary results
# SUMMARY_DIR = './results/tf_ddpg'
# RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(DQN):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, *args, **kwargs):
        from keras import backend as K
        self.K = K
        self.tf = self.K.tf
        self.sess = self.K.get_session()
        self.tau = 0.001
        super(ActorNetwork, self).__init__(*args, **kwargs)

    def build_model(self):
        self.model = super(ActorNetwork, self).build_model()
        self.target_model = clone_model(self.model)

        self.actor_state = self.model.inputs[0]
        self.out = self.model.output
        self.scaled_out = self.tf.multiply(self.out, self.env_spec['action_bound_high'])
        self.network_params = self.model.trainable_weights

        self.target_actor_state = self.target_model.inputs[0]
        self.target_out = self.target_model.output
        self.target_scaled_out = self.tf.multiply(self.target_out, self.env_spec['action_bound_high'])
        self.target_network_params = self.target_model.trainable_weights

        # Op for updating target network
        self.update_target_network_op = []
        for i, t_w in enumerate(self.target_network_params):
            op = t_w.assign(
                self.tf.multiply(self.tau, self.network_params[i]) + self.tf.multiply(1. - self.tau, t_w))
            self.update_target_network_op.append(op)

        # will be fed as self.action_gradient: critic_grads
        self.action_gradient = self.tf.placeholder(
            self.tf.float32, [None, self.env_spec['action_dim']])

        # final gradients op for actor network
        # TODO need to scale out
        self.actor_gradients = self.tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = self.tf.train.AdamOptimizer(self.lr).apply_gradients(
            zip(self.actor_gradients, self.network_params))
        return self.model

    def compile_model(self):
        pass

    def recompile_model(self, sys_vars):
        pass

    def update(self):
        self.sess.run(self.update_target_network_op)

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.actor_state: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.actor_state: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_actor_state: inputs
        })


class CriticNetwork(DQN):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, *args, **kwargs):
        from keras.layers import Dense, Merge
        from keras import backend as K
        self.Merge = Merge
        self.K = K
        self.tf = self.K.tf
        self.sess = self.K.get_session()
        self.tau = 0.001
        super(CriticNetwork, self).__init__(*args, **kwargs)


    def build_critic_models(self):
        state_branch = self.Sequential()
        state_branch.add(self.Dense(
            self.hidden_layers[0],
            input_shape=(self.env_spec['state_dim'],),
            activation=self.hidden_layers_activation,
            init='lecun_uniform'))

        action_branch = self.Sequential()
        action_branch.add(self.Dense(
            self.hidden_layers[0],
            input_shape=(self.env_spec['action_dim'],),
            activation=self.hidden_layers_activation,
            init='lecun_uniform'))

        input_layer = self.Merge([state_branch, action_branch], mode='concat')

        model = self.Sequential()
        model.add(input_layer)

        if (len(self.hidden_layers) > 1):
            for i in range(1, len(self.hidden_layers)):
                model.add(self.Dense(
                    self.hidden_layers[i],
                    init='lecun_uniform',
                    activation=self.hidden_layers_activation))

        model.add(self.Dense(1,
                             init='lecun_uniform',
                             activation='linear'))
        logger.info('Critic model summary')
        model.summary()
        self.model = model

        logger.info("Model built")
        return self.model


    def mean_squared_error(self, y_true, y_pred):
        return self.K.mean(self.K.square(y_pred - y_true), axis=-1)

    def build_model(self):
        self.model = self.build_critic_models()
        self.target_model = clone_model(self.model)

        self.critic_state = self.model.inputs[0]
        self.critic_action = self.model.inputs[1]
        self.out = self.model.output
        self.network_params = self.model.trainable_weights

        self.target_critic_state = self.target_model.inputs[0]
        self.target_critic_action = self.target_model.inputs[1]
        self.target_out = self.target_model.output
        self.target_network_params = self.target_model.trainable_weights

        # Op for updating target network
        self.update_target_network_op = []
        for i, t_w in enumerate(self.target_network_params):
            op = t_w.assign(
                self.tf.multiply(self.tau, self.network_params[i]) + self.tf.multiply(1. - self.tau, t_w))
            self.update_target_network_op.append(op)

        # custom loss and optimization Op
        self.q_prime = self.tf.placeholder(self.tf.float32, [None, 1])
        self.loss = self.mean_squared_error(self.q_prime, self.out)
        self.optimize = self.tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradient = self.tf.gradients(self.out, self.critic_action)
        return self.model


    def train(self, inputs, action, q_prime):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.critic_state: inputs,
            self.critic_action: action,
            self.q_prime: q_prime
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.critic_state: inputs,
            self.critic_action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_critic_state: inputs,
            self.target_critic_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradient, feed_dict={
            self.critic_state: inputs,
            self.critic_action: actions
        })

    def update(self):
        self.sess.run(self.update_target_network_op)


class DDPG2(Agent):

    '''
    The DDPG2 agent (algo), from https://arxiv.org/abs/1509.02971
    reference: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
    https://github.com/matthiasplappert/keras-rl
    '''

    def __init__(self, *args, **kwargs):
        # import only when needed to contain side-effects
        # from keras.layers import Dense, Merge
        # from keras.models import Sequential
        from keras import backend as K
        # self.Dense = Dense
        # self.Merge = Merge
        # self.Sequential = Sequential
        self.K = K
        self.sess = self.K.get_session()

        self.epi = 0
        self.n_epoch = 1
        self.batch_size = 64
        self.gamma = 0.99

        # self.TAU = 0.001  # for target network updates
        super(DDPG2, self).__init__(*args, **kwargs)
        self.build_model(*args, **kwargs)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, *args, **kwargs):
        self.actor = ActorNetwork(*args, **kwargs)
        self.critic = CriticNetwork(*args, **kwargs)

    def compile_model(self):
        pass

    def select_action(self, state):
        i = self.epi
        action = self.actor.predict(np.reshape(
            state, (-1, self.env_spec['state_dim']))) + (1. / (1. + i))
        # print('action shape')
        # print('action shape')
        # print('action shape')
        # print(action)
        # print(action.shape)
        return action[0]

    def update(self, sys_vars):
        self.epi = sys_vars['epi']
        # Update target networks
        self.actor.update()
        self.critic.update()
        return

    def to_train(self, sys_vars):
        return self.memory.size() > MINIBATCH_SIZE
        # return True

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        s_batch = minibatch['states']
        a_batch = minibatch['actions']
        s2_batch = minibatch['next_states']
        # s_batch = np.reshape(minibatch['states'], (-1, self.env_spec['state_dim']))
        # a_batch = np.reshape(minibatch['actions'], (-1, self.env_spec['action_dim']))
        # s2_batch = np.reshape(minibatch['next_states'], (-1, self.env_spec['state_dim']))

        target_q = self.critic.predict_target(
            s2_batch,
            self.actor.predict_target(s2_batch))

        y = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * np.reshape(target_q, (-1))
        y = np.reshape(y, (-1, 1))

        predicted_q_value, _ = self.critic.train(
            s_batch, a_batch, y)
        # minibatch['states'],
        # minibatch['actions'],
        # y)
        # # np.reshape(y, (self.batch_size, 1)))

        # ep_ave_max_q = np.amax(predicted_q_value)
        # print('epi: ' + str(self.epi) + '   Q_max: '+str(ep_ave_max_q))

        # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(s_batch)
        grads = self.critic.action_gradients(s_batch, a_outs)
        self.actor.train(s_batch, grads[0])
        # return actor_loss
        return

        # (Q_states, _states, Q_next_states_max) = self.compute_Q_states(
        #     minibatch)
        # Q_targets = self.compute_Q_targets(
        #     minibatch, Q_states, Q_next_states_max)

        # loss = self.model.train_on_batch(minibatch['states'], Q_targets)

        # errors = abs(np.sum(Q_states  - Q_targets, axis=1))
        # self.memory.update(errors)
        # return loss

    def train(self, sys_vars):
        '''
        Training is for the Q function (NN) only
        otherwise (e.g. policy) see self.update()
        step 1,2,3,4 of algo.
        '''
        loss_total = 0
        for _epoch in range(self.n_epoch):
            loss = self.train_an_epoch()
            # loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss
