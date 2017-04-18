import numpy as np
from rl.util import logger, log_self
from rl.policy.base_policy import Policy


class NoNoisePolicy(Policy):

    '''
    The base class for noise policy for DDPG
    default is no noise
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        super(NoNoisePolicy, self).__init__(env_spec)
        log_self(self)

    def sample(self):
        '''implement noise here, default is none'''
        return 0

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        if self.env_spec['actions'] == 'continuous':
            action = agent.actor.predict(state)[0] + self.sample()
        else:
            Q_state = agent.actor.predict(state)[0]
            assert Q_state.ndim == 1
            action = np.argmax(Q_state)
        return action

    def update(self, sys_vars):
        pass


class LinearNoisePolicy(NoNoisePolicy):

    '''
    policy with linearly decaying noise (1. / (1. + self.epi))
    '''

    def __init__(self, env_spec, exploration_anneal_episodes=20,
                 **kwargs):  # absorb generic param without breaking
        super(LinearNoisePolicy, self).__init__(env_spec)
        self.exploration_anneal_episodes = exploration_anneal_episodes
        self.n_step = 0  # init
        log_self(self)

    def sample(self):
        noise = (1. / (1. + self.n_step))
        return noise

    def update(self, sys_vars):
        epi = sys_vars['epi']
        if epi >= self.exploration_anneal_episodes:
            self.n_step = np.inf  # noise divide to zero
        else:
            self.n_step = sys_vars['epi']


class AnnealedGaussian(LinearNoisePolicy):

    '''
    Base class of random noise policy for DDPG
    Adopted from
    https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py
    '''

    def __init__(self, env_spec, exploration_anneal_episodes,
                 mu, sigma, sigma_min,
                 **kwargs):  # absorb generic param without breaking
        super(AnnealedGaussian, self).__init__(
            env_spec, exploration_anneal_episodes)
        self.size = env_spec['action_dim']
        self.mu = mu
        self.sigma = sigma

        if sigma_min is not None:
            self.m = -(sigma - sigma_min) / self.exploration_anneal_episodes
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * self.n_step + self.c)
        return sigma


class GaussianWhiteNoise(AnnealedGaussian):

    def __init__(self, env_spec, exploration_anneal_episodes=20,
                 mu=0., sigma=.3, sigma_min=None,
                 **kwargs):  # absorb generic param without breaking
        super(GaussianWhiteNoise, self).__init__(
            env_spec, exploration_anneal_episodes,
            mu, sigma, sigma_min)

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        return sample


class OUNoise(AnnealedGaussian):

    '''
    Based on
    http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    '''

    def __init__(self, env_spec, exploration_anneal_episodes=20,
                 theta=.15, mu=0., sigma=.3, dt=1e-2, x0=None, sigma_min=None,
                 **kwargs):  # absorb generic param without breaking
        super(OUNoise, self).__init__(
            env_spec, exploration_anneal_episodes,
            mu, sigma, sigma_min,
            **kwargs)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset_states()

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        x = self.x_prev + self.theta * \
            (self.mu - self.x_prev) * self.dt + self.current_sigma * \
            np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x
