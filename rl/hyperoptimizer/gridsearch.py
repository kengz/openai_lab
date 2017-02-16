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

    def termination_check(self):
        return not (self.next_param_idx < len(self.param_search_list))

    # # generate param_space for hyperopt from experiment_spec
    # def generate_param_space(self):
    #     if self.line_search:
    #         param_grid = param_line_search(self.experiment_spec)
    #     else:
    #         param_grid = param_product(self.experiment_spec)
    #     self.param_space = generate_experiment_spec_grid(
    #         self.experiment_spec, param_grid)
    #     self.num_of_trials = len(self.param_space)

    #     self.trial_array = []
    #     for e in range(self.num_of_trials):
    #         experiment_spec = self.param_space[e]
    #         trial = self.Trial(
    #             experiment_spec, times=self.times, trial_num=e,
    #             num_of_trials=self.num_of_trials,
    #             run_timestamp=self.run_timestamp,
    #             experiment_id_override=self.experiment_id_override)
    #         self.trial_array.append(trial)

    #     return self.param_space

    # @classmethod
    # def mp_run_helper(cls, trial):
    #     return trial.run()

    def algo_step(self):
        self.search()
        self.run_trial()
        self.update_search()

    def run(self):
        # hmm do a spawn style in a loop with termination control,
        # better than pool over array
        return

    def run(self):
        p = mp.Pool(PARALLEL_PROCESS_NUM)
        experiment_data = list(
            p.map(self.mp_run_helper, self.trial_array))
        p.close()
        p.join()
        return experiment_data
