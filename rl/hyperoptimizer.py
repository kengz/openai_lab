import multiprocessing as mp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from rl.util import *


class HyperOptimizer(object):

    '''
    The base class of hyperparam optimizer, with core methods
    '''

    def __init__(self, Experiment, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'sess_spec',
            'times'
        ]
        self.check_set_keys(**kwargs)
        self.run_timestamp = timestamp()
        self.Experiment = Experiment
        self.generate_param_space()

    def check_set_keys(self, **kwargs):
        assert all(k in kwargs for k in self.REQUIRED_GLOBAL_VARS)
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def generate_param_space(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class HyperoptHyperOptimizer(HyperOptimizer):

    def check_set_keys(self, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'sess_spec',
            'times',
            'max_evals'
        ]
        raw_sess_spec = kwargs.pop('sess_spec')
        assert 'param' in raw_sess_spec
        assert 'param_range' in raw_sess_spec
        self.common_sess_spec = copy.deepcopy(raw_sess_spec)
        self.common_sess_spec.pop('param')
        self.common_sess_spec.pop('param_range')
        self.default_param = raw_sess_spec['param']
        self.param_range = raw_sess_spec['param_range']
        self.experiment_num = 0
        self.algo = tpe.suggest

        super(HyperoptHyperOptimizer, self).check_set_keys(**kwargs)

    def convert_to_hp(self, k, v):
        '''
        convert to hyperopt param expressions. refer:
        https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions
        param = {
            'learning_rate': {
                'uniform': {
                    'low': 0.0001,
                    'high': 1.0
                }
            },
            'hidden_layers_activation': ['relu', 'linear']
        }
        for k in param:
            v = param[k]
            print(convert_to_hp(k, v))
        '''
        if isinstance(v, list):
            return hp.choice(k, v)
        elif isinstance(v, dict):
            space_keys = list(v.keys())
            assert len(space_keys) == 1
            space_k = space_keys[0]
            space_v = v[space_k]
            space = getattr(hp, space_k)(k, **space_v)
            return space
        else:
            raise TypeError(
                'sess_spec param_range value must be a list or dict')

    # generate param_space for hyperopt from sess_spec
    def generate_param_space(self):
        self.param_space = copy.copy(self.default_param)
        for k in self.param_range:
            v = self.param_range[k]
            space = self.convert_to_hp(k, v)
            self.param_space[k] = space
        return self.param_space

    def increment_var(self):
        self.experiment_num += 1

    def get_next_var(self):
        self.increment_var()
        return self.__dict__

    def hyperopt_run_experiment(self, param):
        # use param to carry those params other than sess_spec
        # set a global gvs: global variable source
        gv = self.get_next_var()
        sess_spec = gv['common_sess_spec']
        sess_spec.update({'param': param})

        experiment = self.Experiment(
            sess_spec,
            times=gv['times'],
            experiment_num=gv['experiment_num'],
            num_of_experiments=gv['max_evals'],
            run_timestamp=gv['run_timestamp'])
        experiment_data = experiment.run()
        metrics = experiment_data['summary']['metrics']
        # to maximize avg mean rewards/epi via minimization
        hyperopt_loss = -1. * metrics['mean_rewards_per_epi_stats'][
            'mean'] / experiment_data['sys_vars_array'][0][
            'SOLVED_MEAN_REWARD']
        return {'loss': hyperopt_loss,
                'status': STATUS_OK,
                'experiment_data': experiment_data}

    def run(self):
        trials = Trials()
        fmin(fn=self.hyperopt_run_experiment,
             space=self.param_space,
             algo=self.algo,
             max_evals=self.max_evals,
             trials=trials)
        experiment_data_array = [
            trial['result']['experiment_data'] for trial in trials]
        return experiment_data_array


class BruteHyperOptimizer(HyperOptimizer):

    def check_set_keys(self, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'sess_spec',
            'times',
            'line_search'
        ]
        super(BruteHyperOptimizer, self).check_set_keys(**kwargs)

    # generate param_space for hyperopt from sess_spec
    def generate_param_space(self):
        if self.line_search:
            param_grid = param_line_search(self.sess_spec)
        else:
            param_grid = param_product(self.sess_spec)
        self.param_space = generate_sess_spec_grid(self.sess_spec, param_grid)
        self.num_of_experiments = len(self.param_space)

        self.experiment_array = []
        for e in range(self.num_of_experiments):
            sess_spec = self.param_space[e]
            experiment = self.Experiment(
                sess_spec, times=self.times, experiment_num=e,
                num_of_experiments=self.num_of_experiments,
                run_timestamp=self.run_timestamp)
            self.experiment_array.append(experiment)

        return self.param_space

    def mp_run_helper(self, experiment):
        return experiment.run()

    def run(self):
        p = mp.Pool(PARALLEL_PROCESS_NUM)
        experiment_data_array = list(
            p.map(self.mp_run_helper, self.experiment_array))
        p.close()
        p.join()
        return experiment_data_array
