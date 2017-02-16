from rl.util import *


class HyperOptimizer(object):

    '''
    The base class of hyperparam optimizer, with core methods
    read about it on the documentation
    input: Trial (and some specs), param space P (as standardized specs)
    Algo:
    1. search the next p in P using its internal search algo
    2. run a (slow) function Trial(p) = score (inside trial data)
    3. update search using feedback score
    4. repeat till max steps or fitness condition met

    it will be ran by the experiment as:
    hyperopt = HyperOptimizer(Trial, **experiment_kwargs)
    experiment_data = hyperopt.run()
    '''

    def __init__(self, Trial, **kwargs):
        self.Trial = Trial
        self.REQUIRED_ARGS = [
            'experiment_spec',
            'times'
        ]
        self.set_keys(**kwargs)
        self.init_search()

    def set_keys(self, **kwargs):
        self.run_timestamp = timestamp()
        assert all(k in kwargs for k in self.REQUIRED_ARGS)
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def init_search(self):
        '''initialize the search algo and the search space'''
        raise NotImplementedError()

    def search(self):
        '''algo step 1, search and return the next p for Trial(p)'''
        raise NotImplementedError()

    def run_trial(self, param):
        '''algo step 2, construct and run Trial with the next param'''
        raise NotImplementedError()

    def update_search(self, score):
        '''algo step 3, update search algo using score'''
        raise NotImplementedError()

    def terminate_search(self):
        '''algo step 4, terminate when at max steps or fitness condition met'''
        raise NotImplementedError()

    def run(self):
        '''
        top level method to run the entire hyperoptimizer
        will gather and compose experiment_data, then return it
        '''
        raise NotImplementedError()
