import numpy as np
from rl.agent.base_agent import Agent
from rl.util import logger, log_self, clone_model


class RandomProcess(object):

    def reset_states(self):
        pass


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

    def __init__(self):
        # import only when needed to contain side-effects
        from keras.layers import Dense
        from keras.models import Sequential
        from keras import backend as K
        self.Dense = Dense
        self.Sequential = Sequential
        self.K = K
        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.env_spec['action_dim'], theta=.15, mu=0., sigma=.3)

        # super(DQN, self).__init__(env_spec)

    def create_network(self):
        model = self.Sequential()
        # self.build_hidden_layers(model)
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

    def select_action(self, state):
        action = self.actor.predict(state) + self.random_process.sample()
        return action

        # # Apply noise, if a random process is set.
        # if self.training and self.random_process is not None:
        #     noise = self.random_process.sample()
        #     assert noise.shape == action.shape
        #     action += noise

        # return action

    def compute_Q_states(self, minibatch):
        # note the computed values below are batched in array
        Q_states = np.clip(self.model.predict(minibatch['states']),
                           -self.clip_val, self.clip_val)
        Q_next_states = np.clip(self.model.predict(minibatch['next_states']),
                                -self.clip_val, self.clip_val)
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        return (Q_states, Q_next_states, Q_next_states_max)

    def compute_Q_targets(self, minibatch, Q_states, Q_next_states_max):
        # make future reward 0 if exp is terminal
        Q_targets_a = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_next_states_max
        # set batch Q_targets of a as above, the rest as is
        # minibatch['actions'] is one-hot encoded
        Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
            (1 - minibatch['actions']) * Q_states
        return Q_targets

    def custom_loss(self, y_true, y_pred):
        return self.K.mean(self.K.square(y_pred - y_true))

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        # temp
        mu_prime = self.target_actor.predict(minibatch['next_states'])
        Q_prime = self.target_critic.predict(
            minibatch['next_states'] + mu_prime)
        y = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_prime
        # (Q_states, _states, Q_next_states_max) = self.compute_Q_states(
        # minibatch)
        # Q_targets = self.compute_Q_targets(
        #     minibatch, Q_states, Q_next_states_max)

        if K.backend() == 'tensorflow':
            grads = K.gradients(combined_output, self.actor.trainable_weights)
            grads = [g / float(self.batch_size) for g in grads]  # since TF sums over the batch
        else:
            import theano.tensor as T
            grads = T.jacobian(combined_output.flatten(), self.actor.trainable_weights)
            grads = [K.mean(g, axis=0) for g in grads]
        # loss = self.model.train_on_batch(minibatch['states'], Q_targets)
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
            loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss


class CriticNetwork(object):

    '''critic: Q(s,a|theta_Q), model(state, action)=single_Q_value'''

    def __init__(self):
        # import only when needed to contain side-effects
        from keras.layers import Dense, Merge
        from keras.models import Sequential
        self.Dense = Dense
        self.Merge = Merge
        self.Sequential = Sequential

        # super(DQN, self).__init__(env_spec)

    def create_network(self):

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
        # self.critic.compile(loss=self.custom_loss)
        self.target_critic = clone_model(self.critic)


class DDPG(Agent):

    '''
    The base class of Agent, with the core methods
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        self.env_spec = env_spec

    def compile(self, memory, optimizer, policy, preprocessor):
        # set 2 way references
        self.memory = memory
        self.optimizer = optimizer
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

    def compile_model(self):
        raise NotImplementedError()

    def select_action(self, state):
        self.policy.select_action(state)
        raise NotImplementedError()

    def update(self, sys_vars):
        '''Agent update apart from training the Q function'''
        self.policy.update(sys_vars)
        raise NotImplementedError()

    def to_train(self, sys_vars):
        raise NotImplementedError()

    def train(self, sys_vars):
        raise NotImplementedError()
