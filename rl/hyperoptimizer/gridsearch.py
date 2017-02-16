import multiprocessing as mp
from collections import OrderedDict
from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer
from rl.util import *


# convert a dict of param ranges into
# a list of cartesian products of param_range
# e.g. {'a': [1,2], 'b': [3]} into
# [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
def param_product(experiment_spec):
    # TODO needs to be order-preserving
    default_param = experiment_spec['param']
    param_range = experiment_spec['param_range']
    ordered_param_range = OrderedDict(sorted(param_range.items()))
    print(ordered_param_range)
    keys = sorted(ordered_param_range.keys())
    range_vals = ordered_param_range.values()
    param_grid = []
    for vals in itertools.product(*range_vals):
        param = copy.deepcopy(default_param)
        param.update(dict(zip(keys, vals)))
        param_grid.append(param)
    return param_grid


class GridSearch(HyperOptimizer):

    def check_set_keys(self, **kwargs):
        self.REQUIRED_ARGS = [
            'experiment_spec',
            'times',
            'line_search'
        ]
        super(GridSearch, self).check_set_keys(**kwargs)

    def init_search(self):
        print('init search algo n space')
        print('init search algo n space')
        print('init search algo n space')
        print('init search algo n space')
        print('init search algo n space')
        # note that this is order-preserving, easy
        param_grid = param_product(self.experiment_spec)
        self.param_search_list = generate_experiment_spec_grid(
            self.experiment_spec, param_grid)
        self.num_of_trials = len(self.param_search_list)

    def search(self):
        '''no action needed here for exhaustive trials'''
        return

    def run_trial(self):
        trial_num, experiment_spec = self.next_param()
        trial = self.Trial(
            experiment_spec, trial_num=trial_num,
            times=self.times,
            num_of_trials=self.num_of_trials,
            run_timestamp=self.run_timestamp,
            experiment_id_override=self.experiment_id_override)
        return trial.run()

    def update_search(self):
        '''no action needed here for exhaustive trials'''
        return

    def to_terminate(self):
        return not (self.next_param_idx < len(self.param_search_list))

    def algo_step(self):
        self.search()
        trial_data = self.run_trial()
        self.update_search()
        return trial_data

    def run(self):
        # hmm do a spawn style in a loop with termination control,
        # better than pool over array
        # TODO careful with custom termination method
        pool = mp.Pool(PARALLEL_PROCESS_NUM)
        # maybe we need to externalize (trial_num, param) here since processes
        # are indep
        while (not self.to_terminate()):
            pool.apply_async(
                self.algo_step, callback=self.append_experiment_data)
        pool.close()
        pool.join()
        return self.experiment_data
