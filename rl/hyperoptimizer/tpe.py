from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from rl.util import *


class TPE(HyperOptimizer):

    def check_set_keys(self, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'experiment_spec',
            'times',
            'max_evals'
        ]
        raw_experiment_spec = kwargs.pop('experiment_spec')
        assert 'param' in raw_experiment_spec
        assert 'param_range' in raw_experiment_spec
        self.common_experiment_spec = copy.deepcopy(raw_experiment_spec)
        self.common_experiment_spec.pop('param')
        self.common_experiment_spec.pop('param_range')
        self.default_param = raw_experiment_spec['param']
        self.param_range = raw_experiment_spec['param_range']
        self.trial_num = 0
        self.algo = tpe.suggest

        super(TPE, self).check_set_keys(**kwargs)

    @classmethod
    def convert_to_hp(cls, k, v):
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
                'experiment_spec param_range value must be a list or dict')

    # generate param_space for hyperopt from experiment_spec
    def generate_param_space(self):
        self.param_space = copy.copy(self.default_param)
        for k in self.param_range:
            v = self.param_range[k]
            space = self.convert_to_hp(k, v)
            self.param_space[k] = space
        return self.param_space

    def increment_var(self):
        self.trial_num += 1

    def get_next_var(self):
        self.increment_var()
        return self.__dict__

    def hyperopt_run_trial(self, param):
        # use param to carry those params other than experiment_spec
        # set a global gvs: global variable source
        gv = self.get_next_var()
        experiment_spec = gv['common_experiment_spec']
        experiment_spec.update({'param': param})

        trial = self.Trial(
            experiment_spec,
            times=gv['times'],
            trial_num=gv['trial_num'],
            num_of_trials=gv['max_evals'],
            run_timestamp=gv['run_timestamp'])
        trial_data = trial.run()
        metrics = trial_data['summary']['metrics']
        # to maximize avg mean rewards/epi via minimization
        hyperopt_loss = -1. * metrics['mean_rewards_per_epi_stats'][
            'mean'] / trial_data['sys_vars_array'][0][
            'SOLVED_MEAN_REWARD']
        return {'loss': hyperopt_loss,
                'status': STATUS_OK,
                'trial_data': trial_data}

    def run(self):
        trials = Trials()
        fmin(fn=self.hyperopt_run_trial,
             space=self.param_space,
             algo=self.algo,
             max_evals=self.max_evals,
             trials=trials)
        experiment_data = [
            trial['result']['trial_data'] for trial in trials]
        return experiment_data
