class Optimizer(object):

    '''
    The base class of Optimizer, with the core methods
    '''

    def __init__(self, param):
        '''Construct externally, and set at Agent.compile()'''
        self.agent = None
        self.param = param

    def set_init_optim_params(self):
        raise NotImplementedError()

    def init_optimizer(self):
        raise NotImplementedError()

    def change_optim_params(self, new_params):
        # params is a dict containing different settings
        # which depend on the optimizer used
        raise NotImplementedError()