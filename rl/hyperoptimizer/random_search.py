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

    # calculate the constant radius needed to traverse unit cube
    def unit_cube_traversal_radius(self):
        traversal_diameter = 1/np.power(self.max_evals,
                                        1/self.param_space_n_dim)
        traversal_radius = traversal_diameter/2
        return traversal_radius

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
        self.param_space_n_dim = len(self.param_range_keys)
        # careful with resuming, reread shits from search history
        # self.default_param
        # self.param_range
        # iterate, if is dict do init_sampler
        # self.sampler = {
        #     'lr':
        # }
        return

    def search(self):
        '''
        algo:
        1. init x a random position in space
        2. until termination (max_eval or fitness, e.g. solved all), do:
            2.1 sample new pos some radius away: y = x + r
            2.2 if f(y) > f(x) then set x = y

        * Careful, we always do maximization,
        '''
        return

    def decay_radius(self):
        '''future implementation, start of half cube for diameter (so 1/4 for radius), then decay'''
        return

    def update_search(self):
        '''
        to save an restore experiment:
        - all searched points
        - all updated points and their fitness score, just use performance_score, current pointer of x as last
        careful with param_search_list and next_param_idx
        need to get stuff from self.experiment_data
        path:
        # assert self.experiment_data non empty
        last_trial_num = len(self.experiment_data) - 1
        last_trial_data = self.experiment_data[-1]
        param = last_trial_data['experiment_spec']['param']
        metrics = last_trial_data['metrics']
        fitness_score = metrics['fitness_score']

        '''
        return

    def satisfy_fitness(self):
        '''use performance score, solved ratio, solved mean reward'''
        # ideal_fitness_score = util.PROBLEMS: SOLVED_MEAN_REWARD/MAX_EPISODES/2
        return

    def to_terminate(self):
        return
