# The experiment logic and analysis
import json
import multiprocessing as mp
from functools import partial
from rl.session import Session
from rl.spec import game_specs
from rl.util import *


class Experiment(object):

    '''
    The experiment class for each unique sess_spec
    runs for a specified number of times
    Requirements:
    JSON, single file, quick and useful summary,
    replottable data, rerunnable specs
    Keys:
    all below X array of hyper param selection:
    - sess_spec (so we can plug in directly again to rerun)
    - summary
        - time_start
        - time_end
        - time_taken
        - metrics
    - sys_vars_array
    '''

    def __init__(self, sess_spec, times=1):
        self.sess_spec = sess_spec
        self.data_grid = []
        self.times = times
        self.sess_spec.pop('param_range', None)  # single exp, del range

    def analyze(self, data):
        '''
        helper: analyze given data from an experiment
        return metrics
        '''
        sys_vars_array = data['sys_vars_array']
        mean_r_array = [sys_vars['mean_rewards']
                        for sys_vars in sys_vars_array]
        metrics = {
            'experiment_mean': np.mean(mean_r_array),
            'experiment_std': np.std(mean_r_array),
        }
        return metrics

    def save(self):
        '''
        save the entire experiment data grid from inside run()
        '''
        # sort data, best first
        self.data_grid.sort(
            key=lambda data: data['summary']['metrics']['experiment_mean'],
            reverse=True)
        sample_spec = stringify_param(self.sess_spec)
        filename = './data/{}_{}_{}_{}_{}.json'.format(
            sample_spec['problem'],
            sample_spec['Agent'],
            sample_spec['Memory'],
            sample_spec['Policy'],
            timestamp()
        )
        with open(filename, 'w') as f:
            f.write(to_json(self.data_grid))
        logger.info('Experiment complete, written to {}'.format(filename))

    def run(self):
        '''
        helper: run a experiment for Session
        a number of times times given a sess_spec from gym_specs
        '''
        time_start = timestamp()
        sess = Session(problem=self.sess_spec['problem'],
                       Agent=self.sess_spec['Agent'],
                       Memory=self.sess_spec['Memory'],
                       Policy=self.sess_spec['Policy'],
                       param=self.sess_spec['param'])
        sys_vars_array = [sess.run() for i in range(self.times)]
        time_end = timestamp()
        time_taken = timestamp_elapse(time_start, time_end)

        data = {  # experiment data
            'sess_spec': stringify_param(self.sess_spec),
            'summary': {
                'time_start': time_start,
                'time_end': time_end,
                'time_taken': time_taken,
                'metrics': None,
            },
            'sys_vars_array': sys_vars_array,
        }

        data['summary'].update({'metrics': self.analyze(data)})
        # progressive update of data_grid, write when an exp is done
        self.data_grid.append(data)
        self.save()
        return data


def run(sess_name_or_spec, times=1, param_selection=False):
    '''
    primary method:
    run all experiments, specified by the sess_spec or its name
    for a specified number of times per experiment
    (multiple experiments if param_selection=True)
    '''
    if isinstance(sess_name_or_spec, str):
        sess_spec = game_specs.get(sess_name_or_spec)
    else:
        sess_spec = sess_name_or_spec

    if param_selection:
        raise Exception('to be implemented, with separate py processes')
        # param_grid = param_product(
        #     sess_spec['param'], sess_spec['param_range'])
        # sess_spec_grid = [{
        #     'problem': sess_spec['problem'],
        #     'Agent': sess_spec['Agent'],
        #     'Memory': sess_spec['Memory'],
        #     'Policy': sess_spec['Policy'],
        #     'param': param,
        # } for param in param_grid]
        # p = mp.Pool(mp.cpu_count())
        # list(p.map(
        #     partial(run_single_exp, data_grid=data_grid, times=times),
        #     sess_spec_grid))
    else:
        # run_single_exp(sess_spec, data_grid=data_grid, times=times)
        experiment = Experiment(sess_spec, times=times)
        return experiment.run()
