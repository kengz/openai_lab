class Agent(object):

    '''
    The base class of Agent, with the core methods
    '''

    def __init__(self, env_spec):
        self.env_spec = env_spec

    def compile(self, replay_memory, policy):
        self.replay_memory = replay_memory
        self.policy = policy

    def select_action(self, state):
        self.policy.select_action(state)
        raise NotImplementedError()

    def update(self, sys_vars, replay_memory):
        '''
        Agent update apart from training the Q function
        '''
        self.policy.update(sys_vars, replay_memory)
        raise NotImplementedError()

    def train(self, sys_vars, replay_memory):
        raise NotImplementedError()
