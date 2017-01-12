# The experiment logic and analysis
# Requirements:
# JSON, single file, quick and useful summary,
# replottable data, rerunnable specs
# Keys:
# all below X array of hyper param selection:
# - sess_spec (so we can plug in directly again to rerun)
# - summary
#     - time_start
#     - time_end
#     - time_taken
#     - summary metrics
# - sys_vars_array

from rl.util import timestamp


def analyze(data):
    '''
    helper: analyze given data from an experiment
    return metrics
    '''
    sys_vars_array = data['sys_vars_array']
    mean_r_array = [sys_vars['mean_rewards'] for sys_vars in sys_vars_array]
    metrics = {
        'experiment_mean': np.mean(mean_r_array),
        'experiment_std': np.std(mean_r_array),
    }
    return metrics


# TODO
# report speed too (from util)


def save(data_grid):
    '''
    save the entire experiment data grid from inside run()
    '''
    # sort data, best first
    data_grid.sort(
        key=lambda data: data['metrics']['experiment_mean'],
        reverse=True)
    filename = './data/{}_{}_{}_{}_{}.json'.format(
        data_grid[0]['sess_spec']['problem'],
        data_grid[0]['sess_spec']['Agent'],
        data_grid[0]['sess_spec']['Memory'],
        data_grid[0]['sess_spec']['Policy'],
        timestamp()
    )
    with open(filename, 'w') as f:
        json.dump(data_grid, f, indent=2, sort_keys=True)
    logger.info('Experiment complete, written to data/')


def run_single_exp(sess_spec, data_grid, times=1):
    '''
    helper: run a experiment for Session
    a number of times times given a sess_spec from gym_specs
    '''
    start_time = timestamp()
    sess = Session(problem=sess_spec['problem'],
                   Agent=sess_spec['Agent'],
                   Memory=sess_spec['Memory'],
                   Policy=sess_spec['Policy'],
                   param=sess_spec['param'])
    sys_vars_array = [sess.run() for i in range(times)]
    end_time = timestamp()
    data = {  # experiment data
        'start_time': start_time,
        'sess_spec': stringify_param(sess_spec),
        'sys_vars_array': sys_vars_array,
        'metrics': None,
        'end_time': end_time,
    }
    data.update({'metrics': analyze(data)})
    # progressive update of data_grid, write when an exp is done
    data_grid.append(data)
    save(data_grid)
    return data


def run(sess_name, run_param_selection=False, times=1):
    '''
    primary method:
    run the experiment (single or multiple)
    specifying if this should be a param_selection run
    and run each for a number of times
    calls run_single_exp internally
    and employs parallelism whenever possible
    '''
    sess_spec = game_specs.get(sess_name)
    data_grid = []

    if run_param_selection:
        param_grid = param_product(
            sess_spec['param'], sess_spec['param_range'])
        sess_spec_grid = [{
            'problem': sess_spec['problem'],
            'Agent': sess_spec['Agent'],
            'Memory': sess_spec['Memory'],
            'Policy': sess_spec['Policy'],
            'param': param,
        } for param in param_grid]
        p = mp.Pool(mp.cpu_count())
        list(p.map(
            partial(run_single_exp, data_grid=data_grid, times=times),
            sess_spec_grid))
    else:
        run_single_exp(sess_spec, data_grid=data_grid, times=times)

    return data_grid
