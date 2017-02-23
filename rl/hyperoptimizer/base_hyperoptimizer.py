import copy
import multiprocessing as mp
from rl.util import logger, timestamp, PARALLEL_PROCESS_NUM


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
        logger.info('Initialize {}'.format(self.__class__.__name__))
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
        # careful with the key ordering at init_search
        self.default_param = self.experiment_spec['param']
        self.param_range = self.experiment_spec['param_range']

    def compose_experiment_spec(self, param):
        new_experiment_spec = copy.deepcopy(self.experiment_spec)
        new_experiment_spec.pop('param_range', None)
        new_experiment_spec.update({
            'param': param,
        })
        return new_experiment_spec

    def init_search(self):
        '''initialize the search algo and the search space'''
        raise NotImplementedError()

    def search(self):
        '''
        algo step 1, search and return the next p for Trial(p),
        add to (or modify) its internal self.param_search_list
        '''
        raise NotImplementedError()

    def next_param(self):
        '''retrieve trial_num and param, advance the class next_param_idx'''
        assert self.next_param_idx < len(self.param_search_list)
        trial_num = self.next_param_idx
        param = self.param_search_list[self.next_param_idx]
        self.next_param_idx = self.next_param_idx + 1
        return (trial_num, param)

    def run_trial(self, trial_num, param):
        '''
        algo step 2, construct and run Trial with the next param
        '''
        import gc
        experiment_spec = self.compose_experiment_spec(param)
        trial = self.Trial(
            experiment_spec, trial_num=trial_num,
            times=self.times,
            num_of_trials=self.num_of_trials,
            run_timestamp=self.run_timestamp,
            experiment_id_override=self.experiment_id_override)
        trial_data = copy.deepcopy(trial.run())
        del trial
        gc.collect()
        return trial_data

    def update_search(self):
        '''algo step 3, update search algo using self.experiment_data'''
        raise NotImplementedError()

    def to_terminate(self):
        '''algo step 4, terminate when at max steps or fitness condition met'''
        raise NotImplementedError()

    def append_experiment_data(self, trial_data):
        self.experiment_data.append(trial_data)

    def run(self):
        '''
        top level method to run the entire hyperoptimizer
        will gather and compose experiment_data, then return it
        '''
        logger.info('Run {}'.format(self.__class__.__name__))
        pool = mp.Pool(PARALLEL_PROCESS_NUM)
        while (not self.to_terminate()):
            self.search()  # add to self.param_search_list
            trial_num, param = self.next_param()
            pool.apply_async(
                self.run_trial, (trial_num, param),
                callback=self.append_experiment_data)
            self.update_search()
        pool.close()
        pool.join()
        return self.experiment_data
