import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger


class ActorCritic(DQN):

    '''
    Actor Critic algorithm. The actor's policy
    is adjusted in the direction that will lead to
    better actions, guided by the critic
    Implementation adapted from
    http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html

    Assumes one of the policies in actor_critic.py are being used
    '''

    def __init__(self, env_spec,
                 train_per_n_new_exp=1,
                 gamma=0.95, lr=0.1,
                 epi_change_lr=None,
                 batch_size=16, n_epoch=5, hidden_layers=None,
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear',
                 auto_architecture=False,
                 num_hidden_layers=3,
                 first_hidden_layer_size=256,
                 num_initial_channels=16,
                 **kwargs):  # absorb generic param without breaking
        # import only when needed to contain side-effects
        from keras.layers.core import Dense
        from keras.models import Sequential, load_model
        self.Dense = Dense
        self.Sequential = Sequential
        self.load_model = load_model

        super(ActorCritic, self).__init__(env_spec,
                                          train_per_n_new_exp,
                                          gamma, lr,
                                          epi_change_lr,
                                          batch_size, n_epoch, hidden_layers,
                                          hidden_layers_activation,
                                          output_layer_activation,
                                          auto_architecture,
                                          num_hidden_layers,
                                          first_hidden_layer_size,
                                          num_initial_channels,
                                          **kwargs)

    def build_model(self):
        self.build_actor()
        self.build_critic()
        logger.info("Actor and critic models built")

    def build_actor(self):
        actor = self.Sequential()
        super(ActorCritic, self).build_hidden_layers(actor)
        actor.add(self.Dense(self.env_spec['action_dim'],
                             init='lecun_uniform',
                             activation=self.output_layer_activation))
        logger.info("Actor summary")
        actor.summary()
        self.actor = actor

    def build_critic(self):
        critic = self.Sequential()
        super(ActorCritic, self).build_hidden_layers(critic)
        critic.add(self.Dense(1,
                              init='lecun_uniform',
                              activation=self.output_layer_activation))
        logger.info("Critic summary")
        critic.summary()
        self.critic = critic

    def compile_model(self):
        self.actor.compile(
            loss='mse',
            optimizer=self.optimizer.keras_optimizer)
        self.critic.compile(
            loss='mse',
            optimizer=self.optimizer.keras_optimizer)
        logger.info("Actor and critic compiled")

    def recompile_model(self, sys_vars):
        '''
        Option to change model optimizer settings
        Currently only used for changing the learning rate
        Compiling does not affect the model weights
        '''
        if self.epi_change_lr is not None:
            if (sys_vars['epi'] == self.epi_change_lr and
                    sys_vars['t'] == 0):
                self.lr = self.lr / 10.0
                self.optimizer.change_optim_param(**{'lr': self.lr})
                self.actor.compile(
                    loss='mse',
                    optimizer=self.optimizer.keras_optimizer)
                self.critic.compile(
                    loss='mse',
                    optimizer=self.optimizer.keras_optimizer)
                logger.info(
                    'Actor and critic models recompiled with new settings: '
                    'Learning rate: {}'.format(self.lr))

    def train_critic(self, minibatch):
        Q_states = np.clip(self.critic.predict(minibatch['states']),
                           -self.clip_val, self.clip_val)
        Q_next_states = np.clip(self.critic.predict(minibatch['next_states']),
                                -self.clip_val, self.clip_val)
        Q_targets = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_next_states.squeeze()
        Q_targets = np.expand_dims(Q_targets, axis=1)

        actor_delta = Q_next_states - Q_states
        loss = self.critic.train_on_batch(minibatch['states'], Q_targets)

        # update memory, needed for PER
        errors = abs(np.sum(Q_states - Q_targets, axis=1))
        # Q size is only 1, from critic
        assert Q_targets.shape == (self.batch_size, 1)
        assert errors.shape == (self.batch_size, )
        self.memory.update(errors)
        return loss, actor_delta

    def train_actor(self, minibatch, actor_delta):
        old_vals = self.actor.predict(minibatch['states'])
        if self.env_spec['actions'] == 'continuous':
            A_targets = np.zeros(
                (actor_delta.shape[0], self.env_spec['action_dim']))
            for j in range(A_targets.shape[1]):
                A_targets[:, j] = actor_delta.squeeze()
        else:
            A_targets = minibatch['actions'] * actor_delta + \
                (1 - minibatch['actions']) * old_vals

        loss = self.actor.train_on_batch(minibatch['states'], A_targets)
        return loss

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        critic_loss, actor_delta = self.train_critic(minibatch)
        actor_loss = self.train_actor(minibatch, actor_delta)
        return critic_loss + actor_loss
