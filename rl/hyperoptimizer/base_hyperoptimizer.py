from rl.util import *


class HyperOptimizer(object):

    '''
    The base class of hyperparam optimizer, with core methods
    '''

    def __init__(self, Trial, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'experiment_spec',
            'times'
        ]
        self.check_set_keys(**kwargs)
        self.run_timestamp = timestamp()
        self.Trial = Trial
        self.generate_param_space()

    def check_set_keys(self, **kwargs):
        assert all(k in kwargs for k in self.REQUIRED_GLOBAL_VARS)
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def generate_param_space(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
