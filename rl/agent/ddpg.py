import numpy as np
from rl.agent.base_agent import Agent
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

    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
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


class ActorNetwork(object):

    '''
    actor: mu(s|theta_mu), like our typical model(state)=action_val (cont)
    '''

    def __init__(self, parent, env_spec, batch_size=16, n_epoch=5,
                 hidden_layers_shape=None,
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear'):
        # import only when needed to contain side-effects
        from keras.layers import Dense
        from keras.models import Sequential
        from keras import backend as K
        self.Dense = Dense
        self.Sequential = Sequential
        self.K = K

        self.parent = parent
        self.env_spec = env_spec
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.hidden_layers = hidden_layers_shape or [4]
        self.hidden_layers_shape = hidden_layers_shape
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation = output_layer_activation
        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.env_spec['action_dim'], theta=.15, mu=0., sigma=.3)

    def build_model(self):
        model = self.Sequential()
        model.add(self.Dense(self.hidden_layers[0],
                             input_shape=(self.env_spec['state_dim'],),
                             activation=self.hidden_layers_activation,
                             init='lecun_uniform'))
        # inner hidden layer: no specification of input shape
        if (len(self.hidden_layers) > 1):
            for i in range(1, len(self.hidden_layers)):
                model.add(self.Dense(
                    self.hidden_layers[i],
                    init='lecun_uniform',
                    activation=self.hidden_layers_activation))
        model.add(self.Dense(self.env_spec['action_dim'],
                             init='lecun_uniform',
                             activation=self.output_layer_activation))

        self.actor = model
        self.target_actor = clone_model(self.actor)

    def compile_model(self):
        self.actor.compile(
            loss='mse',
            optimizer=self.parent.optimizer.actor_keras_optimizer)
        self.target_actor.compile(
            loss='mse',
            optimizer=self.parent.optimizer.target_actor_keras_optimizer)
        logger.info("Actor Models compiled")

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state)[0] + self.random_process.sample()
        print('action')
        print('action')
        print('action')
        print(action)
        return action

    # def compute_Q_states(self, minibatch):
    #     # note the computed values below are batched in array
    #     Q_states = np.clip(self.actor.predict(minibatch['states']),
    #                        -self.clip_val, self.clip_val)
    #     Q_next_states = np.clip(self.actor.predict(minibatch['next_states']),
    #                             -self.clip_val, self.clip_val)
    #     Q_next_states_max = np.amax(Q_next_states, axis=1)
    #     return (Q_states, Q_next_states, Q_next_states_max)

    # def compute_Q_targets(self, minibatch, Q_states, Q_next_states_max):
    #     # make future reward 0 if exp is terminal
    #     Q_targets_a = minibatch['rewards'] + self.gamma * \
    #         (1 - minibatch['terminals']) * Q_next_states_max
    #     # set batch Q_targets of a as above, the rest as is
    #     # minibatch['actions'] is one-hot encoded
    #     Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
    #         (1 - minibatch['actions']) * Q_states
    #     return Q_targets

    def train_an_epoch(self):
        # TODO train actor properly
        # TODO train target_actor
        return 0

    def train(self, sys_vars):
        '''
        Training is for the Q function (NN) only
        otherwise (e.g. policy) see self.update()
        step 1,2,3,4 of algo.
        '''
        loss_total = 0
        for _epoch in range(self.n_epoch):
            loss = self.train_an_epoch()
            loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss


class CriticNetwork(object):

    '''critic: Q(s,a|theta_Q), model(state, action)=single_Q_value'''

    def __init__(self, parent, env_spec, batch_size=16, n_epoch=5,
                 hidden_layers_shape=None,
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear'):
        # import only when needed to contain side-effects
        from keras.layers import Dense, Merge
        from keras.models import Sequential
        from keras import backend as K
        self.Dense = Dense
        self.Merge = Merge
        self.Sequential = Sequential
        self.K = K

        self.parent = parent
        self.env_spec = env_spec
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.hidden_layers = hidden_layers_shape or [4]
        self.hidden_layers_shape = hidden_layers_shape
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation = output_layer_activation

    def build_model(self):
        action_branch = self.Sequential()
        action_branch.add(self.Dense(self.hidden_layers[0],
                                     input_shape=(
                                         self.env_spec['action_dim'],),
                                     activation=self.hidden_layers_activation,
                                     init='lecun_uniform'))

        state_branch = self.Sequential()
        state_branch.add(self.Dense(self.hidden_layers[0],
                                    input_shape=(self.env_spec['state_dim'],),
                                    activation=self.hidden_layers_activation,
                                    init='lecun_uniform'))

        input_layer = self.Merge([action_branch, state_branch], mode='concat')

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

        self.critic = model
        self.target_critic = clone_model(self.critic)

    def custom_critic_loss(self, y_true, y_pred):
        return self.K.mean(self.K.square(y_true - y_pred))

    def compile_model(self):
        self.critic.compile(
            loss=self.custom_critic_loss,
            optimizer=self.parent.optimizer.critic_keras_optimizer)
        self.target_critic.compile(
            loss='mse',
            optimizer=self.parent.optimizer.target_critic_keras_optimizer)
        logger.info("Critic Models compiled")

    def train_an_epoch(self):
        minibatch = self.parent.memory.rand_minibatch(self.batch_size)
        # temp
        mu_prime = self.target_actor.predict(minibatch['next_states'])
        Q_prime = self.target_critic.predict(
            minibatch['next_states'] + mu_prime)
        y = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_prime
        # TODO missing grad
        # (Q_states, _states, Q_next_states_max) = self.compute_Q_states(
        # minibatch)
        # Q_targets = self.compute_Q_targets(
        #     minibatch, Q_states, Q_next_states_max)

        # if K.backend() == 'tensorflow':
        #     grads = K.gradients(combined_output, self.actor.trainable_weights)
        #     grads = [g / float(self.batch_size) for g in grads]  # since TF sums over the batch
        # else:
        #     import theano.tensor as T
        #     grads = T.jacobian(combined_output.flatten(), self.actor.trainable_weights)
        #     grads = [K.mean(g, axis=0) for g in grads]
        loss = self.actor.train_on_batch(minibatch['states'], y)
        # TODO train target_critic properly
        return loss

    def train(self):
        loss_total = 0
        for _epoch in range(self.n_epoch):
            loss = self.train_an_epoch()
            loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss


class DDPG(Agent):

    '''
    The base class of Agent, with the core methods
    '''

    def __init__(self, env_spec,
                 train_per_n_new_exp=1,
                 gamma=0.95, lr=0.1,
                 epi_change_lr=None,
                 batch_size=16, n_epoch=5, hidden_layers_shape=None,
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear',
                 auto_architecture=False,
                 num_hidden_layers=3,
                 size_first_hidden_layer=256,
                 num_initial_channels=16,
                 **kwargs):  # absorb generic param without breaking
        super(DDPG, self).__init__(env_spec)

        self.train_per_n_new_exp = train_per_n_new_exp
        self.gamma = gamma
        self.lr = lr
        self.epi_change_lr = epi_change_lr
        self.batch_size = batch_size
        self.n_epoch = 1
        self.final_n_epoch = n_epoch
        self.hidden_layers = hidden_layers_shape or [4]
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation = output_layer_activation
        self.clip_val = 10000
        self.auto_architecture = auto_architecture
        self.num_hidden_layers = num_hidden_layers
        self.size_first_hidden_layer = size_first_hidden_layer
        self.num_initial_channels = num_initial_channels

        self.actor_network = ActorNetwork(
            self, env_spec, batch_size, n_epoch,
            hidden_layers_shape,
            hidden_layers_activation,
            output_layer_activation)
        self.critic_network = CriticNetwork(
            self, env_spec, batch_size, n_epoch,
            hidden_layers_shape,
            hidden_layers_activation,
            output_layer_activation)
        log_self(self)
        self.build_model()

    def compile(self, memory, optimizer, policy, preprocessor):
        # override
        # set 2 way references
        self.memory = memory
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

        self.policy = policy
        self.preprocessor = preprocessor
        # back references
        setattr(memory, 'agent', self)
        setattr(optimizer, 'agent', self)
        setattr(policy, 'agent', self)
        setattr(preprocessor, 'agent', self)
        self.compile_model()
        logger.info(
            'Compiled:\nAgent, Memory, Optimizer, Policy, '
            'Preprocessor:\n{}'.format(
                ', '.join([comp.__class__.__name__ for comp in
                           [self, memory, optimizer, policy, preprocessor]])
            ))

    def build_model(self):
        self.actor_network.build_model()
        self.critic_network.build_model()

    def compile_model(self):
        self.actor_network.compile_model()
        self.critic_network.compile_model()

    def select_action(self, state):
        self.actor_network.select_action(state)

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

    def train(self, sys_vars):
        self.actor_network.train(sys_vars)
        self.critic_network.train(sys_vars)
