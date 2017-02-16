from collections import OrderedDict
from rl.hyperoptimizer.line_search import LineSearch
from rl.util import *


class GridSearch(LineSearch):

    def init_search(self):
        '''
        convert a dict of param ranges into
        a list of cartesian products of param_range
        e.g. {'a': [1,2], 'b': [3]} into
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
        note that this is order-preserving, as required by design
        '''
        default_param = self.experiment_spec['param']
        param_range = self.experiment_spec['param_range']
        ordered_param_range = OrderedDict(sorted(param_range.items()))
        keys = sorted(ordered_param_range.keys())
        range_vals = ordered_param_range.values()
        self.param_search_list = []
        for vals in itertools.product(*range_vals):
            param = copy.deepcopy(default_param)
            param.update(dict(zip(keys, vals)))
            self.param_search_list.append(param)
        self.num_of_trials = len(self.param_search_list)
