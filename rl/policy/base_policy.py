class Policy(object):

    '''
    The base class of Policy, with the core methods
    '''

    def __init__(self, env_spec,
                 **kwargs):  # absorb generic param without breaking
        '''Construct externally, and set at Agent.compile()'''
        self.env_spec = env_spec
        self.agent = None

    def select_action(self, state):
        raise NotImplementedError()

    def update(self, sys_vars):
        raise NotImplementedError()
