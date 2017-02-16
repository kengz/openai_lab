from rl.util import *


class HyperOptimizer(object):

    '''
    The base class of hyperparam optimizer, with core methods
    read about it on the documentation
    input: Trial (and some specs), param space P (as standardized specs)
    Algo:
    1. search the next p in P using its internal search algo,
    add to its internal `param_search_list`
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
            'experiment_id_override',
            'times'
        ]
        self.set_keys(**kwargs)
        self.init_search()

    def set_keys(self, **kwargs):
        self.run_timestamp = timestamp()
        self.param_search_list = []
        # the index of next param to try in param_search_list
        self.next_param_idx = 0
        self.experiment_data = []
        assert all(k in kwargs for k in self.REQUIRED_ARGS)
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def next_param(self):
        '''retrieve trial_num and param, advance the class next_param_idx'''
        assert self.next_param_idx < len(self.param_search_list)
        trial_num = self.next_param_idx
        param = self.param_search_list[self.next_param_idx]
        self.next_param_idx = self.next_param_idx + 1
        print('NEXT PARAM')
        print('NEXT PARAM')
        print('NEXT PARAM')
        print('NEXT PARAM')
        print('NEXT PARAM')
        print('NEXT PARAM')
        print(trial_num)
        print(self.next_param_idx)
        return (trial_num, param)

    def append_experiment_data(self, trial_data):
        self.experiment_data.append(trial_data)

    def init_search(self):
        '''initialize the search algo and the search space'''
        raise NotImplementedError()

    def search(self):
        '''
        algo step 1, search and return the next p for Trial(p),
        add to (or modify) its internal self.param_search_list
        '''
        raise NotImplementedError()

    def run_trial(self, param):
        '''
        algo step 2, construct and run Trial with the next param

        '''
        raise NotImplementedError()

    def update_search(self, score):
        '''algo step 3, update search algo using score'''
        raise NotImplementedError()

    def to_terminate(self):
        '''algo step 4, terminate when at max steps or fitness condition met'''
        raise NotImplementedError()

    def run(self):
        '''
        top level method to run the entire hyperoptimizer
        will gather and compose experiment_data, then return it
        '''
        raise NotImplementedError()
