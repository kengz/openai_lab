from util import logger


class Agent(object):

    '''
    The base class of Agent, with the core methods
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        self.env_spec = env_spec

    def compile(self, memory, policy):
        # set 2 way references
        self.memory = memory
        self.policy = policy
        # back references
        setattr(memory, 'agent', self)
        setattr(policy, 'agent', self)
        logger.info('Compiled Agent, Memory, Policy')

    def select_action(self, state):
        self.policy.select_action(state)
        raise NotImplementedError()

    def update(self, sys_vars):
        '''
        Agent update apart from training the Q function
        '''
        self.policy.update(sys_vars)
        raise NotImplementedError()

    def to_train(self, sys_vars):
        raise NotImplementedError()

    def train(self, sys_vars):
        raise NotImplementedError()
