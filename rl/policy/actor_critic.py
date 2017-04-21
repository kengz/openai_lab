import numpy as np
from rl.policy.base_policy import Policy
from rl.policy.epsilon_greedy import EpsilonGreedyPolicy
from rl.policy.boltzmann import BoltzmannPolicy
from rl.util import log_self


class ArgmaxPolicy(Policy):

    '''
    The argmax policy for actor critic agents
    Agent takes the action with the highest
    action score
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        super(ArgmaxPolicy, self).__init__(env_spec)
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        A_score = agent.actor.predict(state)[0]  # extract from batch predict
        assert A_score.ndim == 1
        action = np.argmax(A_score)
        return action

    def update(self, sys_vars):
        pass


class ArgmaxEpsGreedyPolicy(EpsilonGreedyPolicy):

    '''
    The DPG  policy for actor critic agents
    With probability e agent samples randomly
    from the action space
    With probability 1 - e agent selects the argmax of
    the actions
    '''

    def __init__(self, env_spec,
                            init_e=1.0, final_e=0.1, exploration_anneal_episodes=30,
                            **kwargs):  # absorb generic param without breaking
        super(ArgmaxEpsGreedyPolicy, self).__init__(env_spec, init_e, final_e, 
                                                                exploration_anneal_episodes)
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        if self.e > np.random.rand():
            action = np.random.choice(agent.env_spec['actions'])
        else:
            state = np.expand_dims(state, axis=0)
            A_score = agent.actor.predict(state)[0]  # extract from batch predict
            assert A_score.ndim == 1
            action = np.argmax(A_score)
        return action

class ArgmaxSoftmaxPolicy(EpsilonGreedyPolicy):

    '''
    The DPG softmax policy for actor critic agents
    With probability e agent samples from softmax 
    distribution over the action space
    With probability 1 - e agent selects the argmax of
    the actions
    '''

    def __init__(self, env_spec,
                            init_e=1.0, final_e=0.1, exploration_anneal_episodes=30,
                            **kwargs):  # absorb generic param without breaking
        super(ArgmaxSoftmaxPolicy, self).__init__(env_spec, init_e, final_e, 
                                                                exploration_anneal_episodes)
        self.clip_val = 500
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        A_score = agent.actor.predict(state)[0]  # extract from batch predict
        assert A_score.ndim == 1
        if self.e > np.random.rand():
            A_score = A_score.astype('float32')  # fix precision nan issue
            A_score = A_score - np.amax(A_score)  # prevent overflow
            exp_values = np.exp(
                np.clip(A_score, -self.clip_val, self.clip_val))
            assert not np.isnan(exp_values).any()
            probs = np.array(exp_values / np.sum(exp_values))
            probs /= probs.sum()  # renormalize to prevent floating pt error
            action = np.random.choice(agent.env_spec['actions'], p=probs)
        else:
            action = np.argmax(A_score)
        return action


class ArgmaxBoltzmannPolicy(BoltzmannPolicy):

    '''
    The DPG softmax policy for actor critic agents
    With probability e agent samples from softmax 
    distribution over the action space
    With probability 1 - e agent selects the argmax of
    the actions
    '''

    def __init__(self, env_spec,
                            init_e=1.0, final_e=0.1, exploration_anneal_episodes_e=20,
                            init_tau=5., final_tau=0.5, exploration_anneal_episodes=20,
                            **kwargs):  # absorb generic param without breaking
        super(ArgmaxBoltzmannPolicy, self).__init__(env_spec, init_e, final_e, 
                                                                exploration_anneal_episodes)
        self.clip_val = 500
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.exploration_anneal_episodes_e = exploration_anneal_episodes_e
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        A_score = agent.actor.predict(state)[0]  # extract from batch predict
        assert A_score.ndim == 1
        if self.e > np.random.rand():   
            A_score = A_score.astype('float64')  # fix precision nan issue
            A_score = A_score - np.amax(A_score)  # prevent overflow
            exp_values = np.exp(A_score / self.tau) + 0.0001  # prevent underflow
            assert not np.isnan(exp_values).any()
            probs = np.array(exp_values / np.sum(exp_values))
            probs /= probs.sum()  # renormalize to prevent floating pt error
            action = np.random.choice(agent.env_spec['actions'], p=probs)
        else:
            action = np.argmax(A_score)
        return action

    def update(self, sys_vars):
        '''strategy to update tau in agent'''
        epi = sys_vars['epi']
        rise = self.final_tau - self.init_tau
        slope = rise / float(self.exploration_anneal_episodes)
        self.tau = max(slope * epi + self.init_tau, self.final_tau)
        rise = self.final_e - self.init_e
        slope = rise / float(self.exploration_anneal_episodes_e)
        self.e = max(slope * epi + self.init_e, self.final_e)
        return self.tau


class SoftmaxPolicy(Policy):

    '''
    The softmax policy for actor critic agents
    Action is drawn from the prob dist generated
    by softmax(acion_scores)
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        super(SoftmaxPolicy, self).__init__(env_spec)
        self.clip_val = 500
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        A_score = agent.actor.predict(state)[0]  # extract from batch predict
        assert A_score.ndim == 1
        A_score = A_score.astype('float32')  # fix precision nan issue
        A_score = A_score - np.amax(A_score)  # prevent overflow
        exp_values = np.exp(
            np.clip(A_score, -self.clip_val, self.clip_val))
        assert not np.isnan(exp_values).any()
        probs = np.array(exp_values / np.sum(exp_values))
        probs /= probs.sum()  # renormalize to prevent floating pt error
        action = np.random.choice(agent.env_spec['actions'], p=probs)
        return action

    def update(self, sys_vars):
        pass


class GaussianPolicy(Policy):

    '''
    Continuous policy for actor critic models
    Output of the actor network is the mean action
    along each dimension. Action chosen is the mean
    plus some noise parameterized by the variance
    '''

    def __init__(self, env_spec,
                 variance=1.0,
                 **kwargs):  # absorb generic param without breaking
        super(GaussianPolicy, self).__init__(env_spec)
        self.variance = variance
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        a_mean = agent.actor.predict(state)[0]  # extract from batch predict
        action = a_mean + np.random.normal(
            loc=0.0, scale=self.variance, size=a_mean.shape)
        return action

    def update(self, sys_vars):
        pass


class BoundedPolicy(Policy):

    '''
    The bounded policy for actor critic agents
    and continous, bounded policy spaces
    Action bounded above and below by
    - action_bound, + action_bound
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        super(BoundedPolicy, self).__init__(env_spec)
        self.action_bound = env_spec['action_bound_high']
        assert env_spec['action_bound_high'] == -env_spec['action_bound_low']
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        A_score = agent.actor.predict(state)[0]  # extract from batch predict
        action = np.tanh(A_score) * self.action_bound
        return action

    def update(self, sys_vars):
        pass
