class Agent(object):

    '''
    The base class of Agent, with the core methods
    '''

    def __init__(self, env_spec):
        self.env_spec = env_spec

    def select_action(self, state):
        raise NotImplementedError()

    def train(self, sys_vars, replay_memory):
        raise NotImplementedError()
