import numpy as np


class QLearner(object):

    """docstring for QLearner"""

    def __init__(self, state_dim,
                 num_actions,
                 init_exp=0.5,  # initial exploration prob
                 final_exp=0.0,  # final exploration prob
                 anneal_steps=500,  # N steps for annealing exploration
                 alpha=0.2,
                 discount_factor=0.9):  # power factor for discount future rewards

        # Q learning params
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.exploration = init_exp
        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps
        self.discount_factor = discount_factor
        self.alpha = alpha

        # counters
        self.train_iteration = 0

        # table of Q values
        self.qtable = np.random.uniform(
            low=-1, high=1, size=(state_dim, num_actions))

    def initializeState(self, state):
        self.state = state
        # in this state, sort by higher reward, take the action
        self.action = self.qtable[state].argsort()[-1]
        return self.action

    # select action based on epsilon-greedy exploration
    # with decresing exploration prob
    def eGreedyAction(self, state):
        if self.exploration > np.random.rand():
            action = np.random.randint(0, self.num_actions)
        else:
            action = self.qtable[state].argsort()[-1]
        return action

    def updateModel(self, state, reward):
        action = self.eGreedyAction(state)
        self.train_iteration += 1
        self.annealExploration()
        # update Bellman equation
        self.qtable[self.state, self.action] = (1 - self.alpha) * self.qtable[
            self.state, self.action] + self.alpha * (reward + self.discount_factor * self.qtable[state, action])

        self.state = state
        self.action = action

        return self.action

    def annealExploration(self, strategy='linear'):
        # decrease epsilon prob
        ratio = max(
            (self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
        self.exploration = self.final_exp + \
            ratio * (self.init_exp - self.final_exp)

    def save(self, model_path):
        return np.save(model_path, self.qtable)

    def load(self, model_path):
        self.qtable = np.load(model_path)
        return self.qtable

