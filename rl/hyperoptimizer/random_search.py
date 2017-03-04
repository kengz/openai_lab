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

    @classmethod
    def sample_hypersphere(cls, dim, r=1):
        '''Marsaglia algo for sampling uniformly on a hypersphere'''
        v = np.random.randn(dim)
        v = v * r / np.linalg.norm(v)
        return v

    # biject [0, 1] to [x_min, x_max]
    @classmethod
    def biject_continuous(cls, norm_val, x_min, x_max):
        return norm_val*(x_max - x_min) + x_min

    # biject [0, 1] to x_list = [a, b, c, ...] by binning
    def biject_discrete(self, norm_val, x_list):
        list_len = len(x_list)
        inds = np.arange(list_len)
        cont_val = self.biject_continuous(norm_val, 0, list_len)
        ind = np.digitize(cont_val, inds) - 1
        return x_list[ind]

    def biject_dim(self, norm_val, dim_spec):
        if isinstance(dim_spec, list):  # discrete
            return self.biject_discrete(norm_val, dim_spec)
        else:  # cont
            return self.biject_continuous(
                norm_val, dim_spec['min'], dim_spec['max'])
        return

    # biject a vector on unit cube into param_space
    def biject_unit_cube(self, v):
        param_space_v = {}
        for i, param_key in enumerate(self.param_range_keys):
            dim_spec = self.param_range[param_key]
            param_space_v[param_key] = self.biject_dim(v[i], dim_spec)
        return param_space_v

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
        # TODO check dict, has min max
        self.param_space_dims = len(self.param_range_keys)
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
