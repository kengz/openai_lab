import numpy as np


class Policy(object):

    '''
    The base class of Policy, with the core methods
    Acts as a proxy policy definition,
    still draws parameters from agent to compute
    '''

    def __init__(self, agent):
        '''
        call from Agent.__init__ as:
        self.policy = Policy(self)
        '''
        self.agent = agent

    def select_action(self, state):
        raise NotImplementedError()

    def update(self, sys_vars, replay_memory):
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):

    '''
    The Epsilon-greedy policy
    '''

    def select_action(self, state):
        '''epsilon-greedy method'''
        agent = self.agent
        if agent.e > np.random.rand():
            action = np.random.choice(agent.env_spec['actions'])
        else:
            state = np.reshape(state, (1, state.shape[0]))
            # extract from batch predict
            Q_state = agent.model.predict(state)[0]
            assert Q_state.ndim == 1
            action = np.argmax(Q_state)
        return action

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        agent = self.agent
        epi = sys_vars['epi']
        # mem_size = replay_memory.size()
        rise = agent.final_e - agent.init_e
        slope = rise / float(agent.e_anneal_episodes)
        agent.e = max(slope * epi + agent.init_e, agent.final_e)
        return agent.e


class BoltzmannPolicy(Policy):

    '''
    The Boltzmann policy
    TODO still suffers performance decay after solving it for some epis
    '''

    def select_action(self, state):
        agent = self.agent
        self.tau = agent.e + 0.01  # proxied by e for now
        self.clip = (-500., 500.)
        state = np.reshape(state, (1, state.shape[0]))
        Q_state = agent.model.predict(state)[0]  # extract from batch predict
        Q_state = Q_state.astype('float64')  # precision for prob sum to 1
        assert Q_state.ndim == 1
        exp_values = np.exp(
            np.clip(Q_state / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(agent.env_spec['actions'], p=probs)
        return action

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        agent = self.agent
        epi = sys_vars['epi']
        # mem_size = replay_memory.size()
        rise = agent.final_e - agent.init_e
        slope = rise / float(agent.e_anneal_episodes)
        agent.e = max(slope * epi + agent.init_e, agent.final_e)
        return agent.e


class OscillatingEpsilonGreedyPolicy(EpsilonGreedyPolicy):

    '''
    The epsilon-greedy policy with oscillating epsilon
    periodically agent.e will drop to a fraction of
    the current exploration rate
    '''

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        super(OscillatingEpsilonGreedyPolicy, self).update(
            sys_vars, replay_memory)
        agent = self.agent
        epi = sys_vars['epi']
        if not (epi % 3) and epi > 15:
            # drop to 1/3 of the current exploration rate
            agent.e = max(agent.e/3., agent.final_e)
        return agent.e


class TargetedEpsilonGreedyPolicy(EpsilonGreedyPolicy):

    '''
    switch between active and inactive exploration cycles by
    partial mean rewards and its distance to the target mean rewards
    '''

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        agent = self.agent
        epi = sys_vars['epi']
        SOLVED_MEAN_REWARD = sys_vars['SOLVED_MEAN_REWARD']
        REWARD_MEAN_LEN = sys_vars['REWARD_MEAN_LEN']
        PARTIAL_MEAN_LEN = int(REWARD_MEAN_LEN * 0.20)
        if epi < 1:  # corner case when no total_r_history to avg
            return
        # the partial mean for projection the entire mean
        partial_mean_reward = np.mean(
            sys_vars['total_r_history'][-PARTIAL_MEAN_LEN:])
        # difference to target, and its ratio (1 if denominator is 0)
        min_reward = np.amin(sys_vars['total_r_history'])
        projection_gap = SOLVED_MEAN_REWARD - partial_mean_reward
        worst_gap = SOLVED_MEAN_REWARD - min_reward
        gap_ratio = projection_gap / worst_gap
        envelope = agent.init_e + (agent.final_e - agent.init_e) / 2. * \
            (float(epi)/float(agent.e_anneal_episodes))
        pessimistic_gap_ratio = envelope * min(2 * gap_ratio, 1)
        # if is in odd cycle, and diff is still big, actively explore
        active_exploration_cycle = not bool(
            int(epi/PARTIAL_MEAN_LEN) % 2) and (
            projection_gap > abs(SOLVED_MEAN_REWARD * 0.05))
        agent.e = max(pessimistic_gap_ratio * agent.init_e, agent.final_e)

        if not active_exploration_cycle:
            agent.e = max(agent.e/2., agent.final_e)
        return agent.e
