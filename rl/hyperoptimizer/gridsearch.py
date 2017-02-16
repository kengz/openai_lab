from collections import OrderedDict
from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer
from rl.util import *


class GridSearch(HyperOptimizer):

    def cartesian_product(self):
        '''
        convert a dict of param ranges into
        a list of cartesian products of param_range
        e.g. {'a': [1,2], 'b': [3]} into
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
        '''
        default_param = self.experiment_spec['param']
        param_range = self.experiment_spec['param_range']
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

    def check_set_keys(self, **kwargs):
        self.REQUIRED_ARGS = [
            'experiment_spec',
            'times',
            'line_search'
        ]
        super(GridSearch, self).check_set_keys(**kwargs)

    def init_search(self):
        # note that this is order-preserving, as required by design
        param_grid = self.cartesian_product()
        self.param_search_list = generate_experiment_spec_grid(
            self.experiment_spec, param_grid)
        self.num_of_trials = len(self.param_search_list)

    def search(self):
        '''no action needed here for exhaustive trials'''
        return

    def update_search(self):
        '''no action needed here for exhaustive trials'''
        return

    def to_terminate(self):
        return not (self.next_param_idx < len(self.param_search_list))
