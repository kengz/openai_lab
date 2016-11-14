import argparse
import itertools
import json
import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import pprint
matplotlib.rcParams['backend'] = 'agg' if os.environ.get('CI') else 'TkAgg'
import matplotlib.pyplot as plt
from os import path, environ
from functools import partial


# parse_args to add flag
parser = argparse.ArgumentParser(description='Set flag for functions')
parser.add_argument("-d", "--debug",
                    help="activate debug log",
                    action="store_const",
                    dest="loglevel",
                    const=logging.DEBUG,
                    default=logging.INFO)
parser.add_argument("-b", "--blind",
                    help="dont render graphics",
                    action="store_const",
                    dest="render",
                    const=False,
                    default=True)
args = parser.parse_args([]) if environ.get('CI') else parser.parse_args()


# Goddam python logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.setLevel(args.loglevel)
logger.addHandler(handler)
logger.propagate = False

pp = pprint.PrettyPrinter(indent=2)

plt.rcParams['toolbar'] = 'None'  # mute matplotlib toolbar
plotters = {}  # hash of matplotlib objects for live-plot

PROBLEMS = json.loads(open(
    path.join(path.dirname(__file__), 'asset', 'problems.json')).read())

# the keys need to be implemented by a sys_var
# the constants (capitalized) are problem configs,
# set in asset/problems.json
required_sys_keys = {
    'RENDER',
    'GYM_ENV_NAME',
    'SOLVED_MEAN_REWARD',
    'MAX_TIMESTEPS',
    'MAX_EPISODES',
    'REWARD_MEAN_LEN',
    'param',
    'epi',
    't',
    'history',
    'mean_rewards',
    'total_rewards',
    'solved'
}


def init_sys_vars(problem='CartPole-v0', param={}):
    '''
    init the sys vars for a problem by reading from
    asset/problems.json, then reset the other sys vars
    on reset will add vars: {param, epi, history, mean_rewards, solved}
    '''
    sys_vars = PROBLEMS[problem]
    if (not args.render) or mp.current_process().name != 'MainProcess':
        sys_vars['RENDER'] = False  # mute on parallel
    if environ.get('CI'):
        sys_vars['RENDER'] = False
        sys_vars['MAX_EPISODES'] = 2
    sys_vars['param'] = param
    reset_sys_vars(sys_vars)
    init_plotter(sys_vars)
    return sys_vars


def reset_sys_vars(sys_vars):
    '''reset and check RL system vars before each new session'''
    sys_vars['epi'] = 0
    sys_vars['t'] = 0
    sys_vars['history'] = []
    sys_vars['mean_rewards'] = 0
    sys_vars['total_rewards'] = 0
    sys_vars['solved'] = False
    check_sys_vars(sys_vars)
    return sys_vars


def check_sys_vars(sys_vars):
    '''ensure the requried RL system vars are specified'''
    sys_keys = sys_vars.keys()
    assert all(k in sys_keys for k in required_sys_keys)


def get_env_spec(env):
    '''Helper: return the env specs: dims, actions, reward range'''
    return {
        'state_dim': env.observation_space.shape[0],
        'state_bounds': np.transpose(
            [env.observation_space.low, env.observation_space.high]),
        'action_dim': env.action_space.n,
        'actions': list(range(env.action_space.n)),
        'reward_range': env.reward_range
    }


def report_speed(real_time, total_t):
    '''Report on how fast each time step runs'''
    avg_speed = float(real_time)/float(total_t)
    logger.info('Mean speed: {:.4f} s/step'.format(avg_speed))


def update_history(sys_vars,
                   total_t,
                   total_rewards):
    '''
    update the hisory (list of total rewards)
    max len = REWARD_MEAN_LEN
    then report status
    '''

    sys_vars['history'].append(total_rewards)
    avg_len = sys_vars['REWARD_MEAN_LEN']
    # Calculating mean_reward over last 100 episodes
    mean_rewards = np.mean(sys_vars['history'][-avg_len:])
    solved = (mean_rewards >= sys_vars['SOLVED_MEAN_REWARD'])
    sys_vars['mean_rewards'] = mean_rewards
    sys_vars['total_rewards'] = total_rewards
    sys_vars['solved'] = solved
    live_plot(sys_vars)

    logs = [
        '',
        'Episode: {}, total t: {}, total reward: {}'.format(
            sys_vars['epi'], total_t, total_rewards),
        'Mean rewards over last {} episodes: {:.4f}'.format(
            sys_vars['REWARD_MEAN_LEN'], mean_rewards),
        '{:->20}'.format(''),
    ]
    logger.debug('\n'.join(logs))
    check_session_ends(sys_vars)
    return sys_vars


def check_session_ends(sys_vars):
    if (sys_vars['solved'] or
            (sys_vars['epi'] == sys_vars['MAX_EPISODES'] - 1)):
        logger.info('Problem solved? {}. At epi: {}. Params: {}'.format(
            sys_vars['solved'], sys_vars['epi'],
            pp.pformat(sys_vars['param'])))
    np.savetxt('{}_history.txt'.format(sys_vars['GYM_ENV_NAME']),
               sys_vars['history'], '%.4f', header='total_rewards')
    if not sys_vars['RENDER']:
        return
    plt.savefig('{}.png'.format(sys_vars['GYM_ENV_NAME']))


def init_plotter(sys_vars):
    if not sys_vars['RENDER']:
        return
    # initialize the plotters
    fig = plt.figure(facecolor='white')

    ax1 = fig.add_subplot(211,
                          frame_on=False,
                          title=str(sys_vars['param']) +
                          '\ntotal rewards per episode',
                          ylabel='total rewards')
    p1, = ax1.plot([], [])
    plotters['total rewards'] = (ax1, p1)

    ax2 = fig.add_subplot(212,
                          frame_on=False,
                          title='mean rewards over last 100 episodes',
                          ylabel='mean rewards')
    p2, = ax2.plot([], [])
    plotters['mean rewards'] = (ax2, p2)

    plt.ion()  # for live plot


def live_plot(sys_vars):
    '''do live plotting'''
    if not sys_vars['RENDER']:
        return
    ax1, p1 = plotters['total rewards']
    p1.set_ydata(np.append(p1.get_ydata(), sys_vars['history'][-1]))
    p1.set_xdata(np.arange(len(p1.get_ydata())))
    ax1.relim()
    ax1.autoscale_view(tight=True, scalex=True, scaley=True)

    ax2, p2 = plotters['mean rewards']
    p2.set_ydata(np.append(p2.get_ydata(), sys_vars['mean_rewards']))
    p2.set_xdata(np.arange(len(p2.get_ydata())))
    ax2.relim()
    ax2.autoscale_view(tight=True, scalex=True, scaley=True)

    plt.draw()
    plt.pause(0.01)


# convert a dict of param ranges into
# a list of cartesian products of param_range
# e.g. {'a': [1,2], 'b': [3]} into
# [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
def param_product(param_range):
    keys = param_range.keys()
    range_vals = param_range.values()
    return [dict(zip(keys, vals)) for vals in itertools.product(*range_vals)]


# advanced, experimental code for parallelization
def run_session_average(run_session, problem, param={}):
    '''
    executes the defined run_session function with sys_vars
    run session multiple times for a param
    then average the mean_rewards from them
    '''
    SESSIONS_PER_PARAM = 5
    logger.info(
        'Running average session with param = {}'.format(pp.pformat(param)))
    mean_rewards_history = []
    for i in range(SESSIONS_PER_PARAM):
        sys_vars = run_session(problem, param)
        mean_rewards_history.append(sys_vars['mean_rewards'])
        sessions_mean_rewards = np.mean(mean_rewards_history)
        if sys_vars['solved']:
            break
    logger.info(
        'Sessions mean rewards: {} with param = {}'.format(
            sessions_mean_rewards, pp.pformat(param)))
    return {'param': param, 'sessions_mean_rewards': sessions_mean_rewards}


def select_best_param(run_session, problem, param_grid):
    '''
    Parameter selection
    by running session average for each param parallel
    executes the defined run_session function with problem
    then sort by highest sessions_mean_rewards first
    return the best
    '''
    # NUM_CORES = mp.cpu_count()
    # p = mp.Pool(NUM_CORES)
    # params_means = p.map(
    #     partial(run_session_average, run_session, problem),
    #     param_grid)
    params_means = list(map(
        partial(run_session_average, run_session, problem),
        param_grid))
    params_means.sort(key=lambda pm: pm['sessions_mean_rewards'], reverse=True)
    for pm in params_means:
        logger.info(pp.pformat(pm))
    return params_means[0]
