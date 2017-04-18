import numpy as np
from rl.agent.base_agent import Agent
from rl.agent.dqn import DQN
from rl.util import logger, clone_model, clone_optimizer


class Actor(DQN):
    '''
    Actor of DDPG, with its network and target network
    input is states, output is action
    very similar to DQN
    '''

    def __init__(self, *args, **kwargs):
        from keras import backend as K
        self.K = K
        self.tf = self.K.tf
        self.sess = self.K.get_session()
        self.tau = 0.001
        super(Actor, self).__init__(*args, **kwargs)

    def build_model(self):
        self.model = super(Actor, self).build_model()
        self.target_model = clone_model(self.model)

        self.actor_states = self.model.inputs[0]
        self.out = self.model.output
        self.scaled_out = self.tf.multiply(
            self.out, self.env_spec['action_bound_high'])
        self.network_params = self.model.trainable_weights

        self.target_actor_states = self.target_model.inputs[0]
        self.target_out = self.target_model.output
        self.target_scaled_out = self.tf.multiply(
            self.target_out, self.env_spec['action_bound_high'])
        self.target_network_params = self.target_model.trainable_weights

        # Op for updating target network
        self.update_target_network_op = []
        for i, t_w in enumerate(self.target_network_params):
            op = t_w.assign(
                self.tf.multiply(
                    self.tau, self.network_params[i]
                ) + self.tf.multiply(1. - self.tau, t_w))
            self.update_target_network_op.append(op)

        # will be fed as self.action_gradient: critic_grads
        self.action_gradient = self.tf.placeholder(
            self.tf.float32, [None, self.env_spec['action_dim']])

        # actor model gradient op, to be fed from critic
        self.actor_gradients = self.tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization op
        self.optimize = self.tf.train.AdamOptimizer(self.lr).apply_gradients(
            zip(self.actor_gradients, self.network_params))
        return self.model

    def compile_model(self):
        pass

    def recompile_model(self, sys_vars):
        pass

    def update(self):
        self.sess.run(self.update_target_network_op)

    def predict(self, states):
        return self.sess.run(self.scaled_out, feed_dict={
            self.actor_states: states
        })

    def target_predict(self, next_states):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_actor_states: next_states
        })

    def train(self, states, critic_action_gradient):
        return self.sess.run(self.optimize, feed_dict={
            self.actor_states: states,
            self.action_gradient: critic_action_gradient
        })


class Critic(DQN):

    '''
    Critic of DDPG, with its network and target network
    input is states and actions, output is Q value
    the action is from Actor
    '''

    def __init__(self, *args, **kwargs):
        from keras.layers import Dense, Merge
        from keras import backend as K
        self.Merge = Merge
        self.K = K
        self.tf = self.K.tf
        self.sess = self.K.get_session()
        self.tau = 0.001
        super(Critic, self).__init__(*args, **kwargs)

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

    def build_model(self):
        self.model = self.build_critic_models()
        self.target_model = clone_model(self.model)

        self.critic_states = self.model.inputs[0]
        self.critic_actions = self.model.inputs[1]
        self.out = self.model.output
        self.network_params = self.model.trainable_weights

        self.target_critic_states = self.target_model.inputs[0]
        self.target_critic_actions = self.target_model.inputs[1]
        self.target_out = self.target_model.output
        self.target_network_params = self.target_model.trainable_weights

        # Op for updating target network
        self.update_target_network_op = []
        for i, t_w in enumerate(self.target_network_params):
            op = t_w.assign(
                self.tf.multiply(
                    self.tau, self.network_params[i]
                ) + self.tf.multiply(1. - self.tau, t_w))
            self.update_target_network_op.append(op)

        # custom loss and optimization Op
        self.y = self.tf.placeholder(self.tf.float32, [None, 1])
        self.loss = self.tf.losses.mean_squared_error(self.y, self.out)
        self.optimize = self.tf.train.AdamOptimizer(
            self.lr).minimize(self.loss)

        self.action_gradient = self.tf.gradients(self.out, self.critic_actions)
        return self.model

    def update(self):
        self.sess.run(self.update_target_network_op)

    def get_action_gradient(self, states, actions):
        return self.sess.run(self.action_gradient, feed_dict={
            self.critic_states: states,
            self.critic_actions: actions
        })[0]

    # def predict(self, inputs, action):
    #     return self.sess.run(self.out, feed_dict={
    #         self.critic_states: inputs,
    #         self.critic_actions: action
    #     })

    def target_predict(self, next_states, mu_prime):
        return self.sess.run(self.target_out, feed_dict={
            self.target_critic_states: next_states,
            self.target_critic_actions: mu_prime
        })

    def train(self, states, actions, y):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.critic_states: states,
            self.critic_actions: actions,
            self.y: y
        })


class DDPG2(Agent):

    '''
    DDPG Algorithm, from https://arxiv.org/abs/1509.02971
    has Actor, Critic, and each has its own target network
    Implementation referred from https://github.com/pemami4911/deep-rl
    '''

    def __init__(self, *args, **kwargs):
        # import only when needed to contain side-effects
        from keras import backend as K
        self.K = K
        self.sess = self.K.get_session()

        # TODO absorb properly
        self.epi = 0
        self.n_epoch = 1
        self.batch_size = 64
        self.gamma = 0.99

        super(DDPG2, self).__init__(*args, **kwargs)
        self.build_model(*args, **kwargs)
        self.sess.run(self.K.tf.global_variables_initializer())

    def build_model(self, *args, **kwargs):
        # TODO prolly wanna unify self.tf
        self.actor = Actor(*args, **kwargs)
        self.critic = Critic(*args, **kwargs)

    def compile_model(self):
        pass

    def select_action(self, state):
        # TODO externalize to policy
        i = self.epi
        # TODO can we use expand dims?
        action = self.actor.predict(
            np.expand_dims(state, axis=0)) + (1. / (1. + i))
        return action[0]

    def update(self, sys_vars):
        self.epi = sys_vars['epi']
        # Update target networks
        self.actor.update()
        self.critic.update()

    def to_train(self, sys_vars):
        return self.memory.size() > self.batch_size

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)

        # train critic
        mu_prime = self.actor.target_predict(minibatch['next_states'])
        q_prime = self.critic.target_predict(
            minibatch['next_states'], mu_prime)
        # reshape for element-wise multiplication
        # to feed into network, y shape needs to be (?, 1)
        y = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * np.reshape(q_prime, (-1))
        y = np.reshape(y, (-1, 1))

        # TODO want this to be loss
        predicted_q_value, _, critic_loss = self.critic.train(
            minibatch['states'], minibatch['actions'], y)

        # train actor
        # Update the actor policy using the sampled gradient
        actions = self.actor.predict(minibatch['states'])
        critic_action_gradient = self.critic.get_action_gradient(
            minibatch['states'], actions)
        # TODO want this to be loss too
        actor_loss = self.actor.train(
            minibatch['states'], critic_action_gradient)

        # return actor_loss
        # loss = critic_loss + actor_loss
        return

    def train(self, sys_vars):
        loss_total = 0
        for _epoch in range(self.n_epoch):
            loss = self.train_an_epoch()
            # loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss
