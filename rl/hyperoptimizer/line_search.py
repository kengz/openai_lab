from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer
from rl.util import *


class LineSearch(HyperOptimizer):

    def init_search(self):
        '''
        convert a dict of param ranges into
        a list parameter settings corresponding
        to a line search of the param range
        for each param
        All other parameters set to default vals
        note that this is order-preserving, as required by design
        '''
        default_param = self.experiment_spec['param']
        param_range = self.experiment_spec['param_range']
        keys = sorted(param_range.keys())
        self.param_search_list = []
        for key in keys:
            vals = param_range[key]
            for val in vals:
                param = copy.deepcopy(default_param)
                param[key] = val
                self.param_search_list.append(param)
        self.num_of_trials = len(self.param_search_list)

    def search(self):
        '''no action needed here for exhaustive trials'''
        return

    def update_search(self):
        '''no action needed here for exhaustive trials'''
        return

    def to_terminate(self):
        return not (self.next_param_idx < len(self.param_search_list))
