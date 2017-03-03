import numpy as np
from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer


class RandomSearch(HyperOptimizer):

    def set_keys(self, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'experiment_spec',
            'experiment_id_override',
            'times',
            'max_evals'
        ]
        super(RandomSearch, self).set_keys(**kwargs)

    # def init_sampler(self, real_param):
    #     '''
    #     e.g. for 'lr': {real_param}
    #     where {real_param} = {
    #         "uniform": {
    #             "low": 0,
    #             "high": 1
    #         }
    #     }
    #     '''
    #     keys = list(real_param.keys())
    #     assert len(keys) == 1
    #     dist = keys[0]
    #     dist_kwargs = real_param[dist]
    #     return partial(getattr(np.random, dist), **dist_kwargs)

    def sample_hypersphere(dim, r=1):
        '''Marsaglias algo for sampling uniformly on hypersphere'''
        v = np.random.randn(dim)
        v = v * r / np.linalg.norm(v)
        return v

    def unit_cube_bijection():
        return

    def biject_axis():
        return

    # biject [0, 1] to [x_min, x_max]
    def biject_continuous(norm_val, x_min, x_max):
        return norm_val*(x_max - x_min) + x_min

    # biject [0, 1] to x_list = [a, b, c, ...] by binning
    def biject_discrete(norm_val, x_list):
        x_len = len(x_list)
        inds = np.arange(x_len)
        cont_val = biject_continuous(norm_val, 0, x_len)
        ind = np.digitize(cont_val, inds) - 1
        return x_list[ind]

    def init_search(self):
        '''
        all random space is numpy.random
        specify json by method then args
        e.g. to call numpy.random.uniform(low=0, high=1)
        "uniform": {
            "low": 0,
            "high": 1
        }
        '''
        # self.default_param
        # self.param_range
        # iterate, if is dict do init_sampler
        # self.sampler = {
        #     'lr':
        # }
        return

    def search(self):
        return

    def update_search(self):
        return

    def to_terminate(self):
        return
