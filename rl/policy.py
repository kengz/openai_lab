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
        rise = agent.final_e - agent.init_e
        slope = rise / float(agent.e_anneal_episodes)
        agent.e = max(slope * epi + agent.init_e, agent.final_e)
        return agent.e


class BoltzmannPolicy(Policy):

    '''
    The Boltzmann policy
    '''

    def select_action(self, state):
        agent = self.agent
        # so tau from 5 to 1. seems ok
        # or 5 to 0.2 ++
        agent.init_e = 5.  # init_tau, proxied by e
        agent.final_e = 0.6  # final_tau, proxied by e
        # ok need to clip at 80 / 0.5
        # so divide by max of norm vs
        # if q under 80, ok,
        # else, rescale to fit inside 80
        # /200*80
        self.tau = agent.e  # proxied by e for now
        state = np.reshape(state, (1, state.shape[0]))
        Q_state = agent.model.predict(state)[0]  # extract from batch predict
        assert Q_state.ndim == 1
        # TODO notes: peculiar things
        # q val needs to reach high enough before it can reliably perform, guess it makes sense regarding convergence. typically q > 60
        # also q val cannot be too high, otherwise the exp_values hence the prob goes crazy. typically < 80
        # so, for tau ending at 0.5, 60 < q < 80
        # which after /tau is at 120-160 for np.exp()
        # TODO for generality, the best way may be to just decay tau by max q, so the rescale the np.exp value to a range, without clipping
        # TODO or adjust tau with max_q
        max_q = np.amax(Q_state)
        if (max_q > 80.):
            self.tau = self.tau * 80. / max_q
        if (10. < max_q < 60.): # allow the beginning <10 to explore first
            self.tau = self.tau * max_q / 60.

        Q_state = Q_state.astype('float64')  # prevent precision overflow
        exp_values = np.exp(Q_state / self.tau)
        assert not np.isnan(exp_values).any()
        probs = np.array(exp_values / np.sum(exp_values))
        probs /= probs.sum()  # renormalize for floating pt error
        print(Q_state)
        print(exp_values)
        print(probs)
        print(np.amax(probs))
        action = np.random.choice(agent.env_spec['actions'], p=probs)
        return action

    def update(self, sys_vars, replay_memory):
        '''strategy to update epsilon in agent'''
        agent = self.agent
        epi = sys_vars['epi']
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
