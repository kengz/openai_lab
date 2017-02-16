class Optimizer(object):

    '''
    The base class of Optimizer, with the core methods
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        '''Construct externally, and set at Agent.compile()'''
        self.agent = None
        self.optim_name = optim_name
        self.param = param

    def init_optimizer(self):
        raise NotImplementedError()

    def change_optim_params(self, new_params):
        # params is a dict containing different settings 
        # which depend on the optimizer used
        raise NotImplementedError()