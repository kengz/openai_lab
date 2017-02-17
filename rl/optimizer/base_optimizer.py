from rl.util import log_self


class Optimizer(object):

    '''
    The base class of Optimizer, with the core methods
    '''

    def __init__(self, **kwargs):
        '''Construct externally, and set at Agent.compile()'''
        self.agent = None
        self.keras_optimizer = None
        self.optim_param = None
        self.set_init_optim_param(**kwargs)
        self.init_optimizer()
        log_self(self)

    def set_init_optim_param(self, **kwargs):
        o_param = {k: kwargs.get(k) for k in self.optim_param_keys}
        self.optim_param = {k: v for k, v in o_param.items() if v is not None}

    def init_optimizer(self):
        raise NotImplementedError()

    def change_optim_param(self, **new_param):
        self.set_init_optim_param(**new_param)
        self.init_optimizer()
        logger.info("Optimizer param changed")
        log_self(self)
