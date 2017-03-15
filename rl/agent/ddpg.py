import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger, log_self, clone_model, clone_optimizer


class RandomProcess(object):

    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):

    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on
# http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):

    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None,
                 size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * \
            (self.mu - self.x_prev) * self.dt + self.current_sigma * \
            np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class DDPG(DQN):

    '''
    The base class of Agent, with the core methods
    '''

    def __init__(self, *args, **kwargs):
        # import only when needed to contain side-effects
        from keras.layers import Dense, Merge
        from keras.models import Sequential
        from keras import backend as K
        self.Dense = Dense
        self.Merge = Merge
        self.Sequential = Sequential
        self.K = K

        super(DDPG, self).__init__(*args, **kwargs)
        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.env_spec['action_dim'], theta=.15, mu=0., sigma=.3)

    def compile(self, memory, optimizer, policy, preprocessor):
        # override to make 4 optimizers
        self.optimizer = optimizer
        # clone for actor, critic networks
        self.optimizer.actor_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.optimizer.target_actor_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.optimizer.critic_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.optimizer.target_critic_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        del self.optimizer.keras_optimizer

        super(DDPG, self).compile(memory, self.optimizer, policy, preprocessor)

    def build_actor_models(self):
        model = self.Sequential()
        self.build_hidden_layers(model)
        model.add(self.Dense(self.env_spec['action_dim'],
                             init='lecun_uniform',
                             activation=self.output_layer_activation))
        logger.info('Actor model summary')
        model.summary()
        self.actor = model
        self.target_actor = clone_model(self.actor)

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
                             activation=self.output_layer_activation))
        logger.info('Critic model summary')
        model.summary()
        self.critic = model
        self.target_critic = clone_model(self.critic)

    def build_model(self):
        self.build_actor_models()
        self.build_critic_models()

    def custom_critic_loss(self, y_true, y_pred):
        return self.K.mean(self.K.square(y_true - y_pred))

    def compile_model(self):
        self.actor_state = self.actor.inputs[0]
        self.action_gradient = self.K.placeholder(
            shape=(None, self.env_spec['action_dim']))
        self.actor_grads = self.K.tf.gradients(
            self.actor.output, self.actor.trainable_weights,
            -self.action_gradient)
        self.actor_optimize = self.K.tf.train.AdamOptimizer(
            self.lr).apply_gradients(
            zip(self.actor_grads, self.actor.trainable_weights))

        self.critic_state = self.critic.inputs[0]
        self.critic_action = self.critic.inputs[1]
        self.critic_action_grads = self.K.tf.gradients(
            self.critic.output, self.critic_action)

        # self.actor.compile(
        #     loss='mse',
        #     optimizer=self.optimizer.actor_keras_optimizer)
        self.target_actor.compile(
            loss='mse',
            optimizer=self.optimizer.target_actor_keras_optimizer)
        logger.info("Actor Models compiled")

        self.critic.compile(
            loss=self.custom_critic_loss,
            optimizer=self.optimizer.critic_keras_optimizer)
        self.target_critic.compile(
            loss='mse',
            optimizer=self.optimizer.target_critic_keras_optimizer)
        logger.info("Critic Models compiled")

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state)[0] + self.random_process.sample()
        return action

    def update(self, sys_vars):
        '''Agent update apart from training the Q function'''
        return

    def to_train(self, sys_vars):
        '''
        return boolean condition if agent should train
        get n NEW experiences before training model
        '''
        t = sys_vars['t']
        done = sys_vars['done']
        timestep_limit = self.env_spec['timestep_limit']
        return (t > 0) and bool(
            t % self.train_per_n_new_exp == 0 or
            t == (timestep_limit-1) or
            done)

    def get_trainable_params(self, model):
        import keras
        params = []
        for layer in model.layers:
            params += keras.engine.training.collect_trainable_weights(layer)
        return params

    def update_critic(self, minibatch):
        mu_prime = self.target_actor.predict(minibatch['next_states'])
        Q_prime = self.target_critic.predict(
            [minibatch['next_states'], mu_prime])
        y = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_prime
        critic_loss = self.critic.train_on_batch(
            [minibatch['states'], minibatch['actions']], y)
        return critic_loss

    def update_actor(self, minibatch):
        actions = self.actor.predict(minibatch['states'])
        # critic_grads = critic.gradients(minibatch['states'], actions)
        critic_grads = self.K.get_session().run(
            self.critic_action_grads, feed_dict={
                self.critic_state: minibatch['states'],
                self.critic_action: actions
            })[0]

        # actor.train(minibatch['states'], critic_grads)
        self.K.get_session().run(self.actor_optimize, feed_dict={
            self.actor_state: minibatch['states'],
            self.action_gradient: critic_grads
        })
        actor_loss = 0
        return actor_loss

    def update_target_networks(self):
        self.TAU = 0.01

        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (
                1 - self.TAU) * actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)

        critic_weights = self.critic.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (
                1 - self.TAU) * critic_target_weights[i]
        self.target_critic.set_weights(critic_target_weights)

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        critic_loss = self.update_critic(minibatch)
        actor_loss = self.update_actor(minibatch)
        self.update_target_networks()

        loss = critic_loss + actor_loss
        return loss

    def train(self, sys_vars):
        loss_total = 0
        for _epoch in range(self.n_epoch):
            loss = self.train_an_epoch()
            loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss
