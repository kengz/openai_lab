import numpy as np
from rl.agent.base_agent import Agent
from rl.util import logger, log_self, clone_model


class ActorNetwork(object):

    '''
    actor: mu(s|theta_mu), like our typical model(state)=action_Q_val
    '''

    def __init__(self):
        # import only when needed to contain side-effects
        from keras.layers import Dense
        from keras.models import Sequential
        self.Dense = Dense
        self.Sequential = Sequential

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
