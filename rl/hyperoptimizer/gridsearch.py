import multiprocessing as mp
from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer
from rl.util import *


class GridSearch(HyperOptimizer):

    def check_set_keys(self, **kwargs):
        self.REQUIRED_GLOBAL_VARS = [
            'experiment_spec',
            'times',
            'line_search'
        ]
        super(GridSearch, self).check_set_keys(**kwargs)

    # generate param_space for hyperopt from experiment_spec
    def generate_param_space(self):
        if self.line_search:
            param_grid = param_line_search(self.experiment_spec)
        else:
            param_grid = param_product(self.experiment_spec)
        self.param_space = generate_experiment_spec_grid(
            self.experiment_spec, param_grid)
        self.num_of_trials = len(self.param_space)

        self.trial_array = []
        for e in range(self.num_of_trials):
            experiment_spec = self.param_space[e]
            trial = self.Trial(
                experiment_spec, times=self.times, trial_num=e,
                num_of_trials=self.num_of_trials,
                run_timestamp=self.run_timestamp,
                experiment_id_override=self.experiment_id_override)
            self.trial_array.append(trial)

        return self.param_space

    @classmethod
    def mp_run_helper(cls, trial):
        return trial.run()

    def run(self):
        p = mp.Pool(PARALLEL_PROCESS_NUM)
        experiment_data = list(
            p.map(self.mp_run_helper, self.trial_array))
        p.close()
        p.join()
        return experiment_data
