import argparse
import copy
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
from datetime import datetime
from os import path, environ


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

# the keys and their defaults need to be implemented by a sys_var
# the constants (capitalized) are problem configs,
# set in asset/problems.json
required_sys_keys = {
    'RENDER': None,
    'GYM_ENV_NAME': None,
    'SOLVED_MEAN_REWARD': None,
    'MAX_EPISODES': None,
    'REWARD_MEAN_LEN': None,
    'PARAM': None,
    'epi': 0,
    't': 0,
    'loss': [],
    'total_r_history': [],
    'explore_history': [],
    'mean_rewards': 0,
    'total_rewards': 0,
    'solved': False,
}


def init_sys_vars(problem='CartPole-v0', param={}):
    '''
    init the sys vars for a problem by reading from
    asset/problems.json, then reset the other sys vars
    on reset will add vars (lower cases, see required_sys_keys)
    '''
    sys_vars = PROBLEMS[problem]
    if (not args.render) or mp.current_process().name != 'MainProcess':
        sys_vars['RENDER'] = False  # mute on parallel
    if environ.get('CI'):
        sys_vars['RENDER'] = False
        sys_vars['MAX_EPISODES'] = 2
    sys_vars['PARAM'] = param
    reset_sys_vars(sys_vars)
    init_plotter(sys_vars)
    return sys_vars


def reset_sys_vars(sys_vars):
    '''reset and check RL system vars (lower case) before each new session'''
    for k in required_sys_keys:
        if k.islower():
            sys_vars[k] = copy.copy(required_sys_keys.get(k))
    check_sys_vars(sys_vars)
    return sys_vars


def check_sys_vars(sys_vars):
    '''ensure the requried RL system vars are specified'''
    sys_keys = sys_vars.keys()
    assert all(k in sys_keys for k in required_sys_keys)


def get_env_spec(env):
    '''Helper: return the env specs: dims, actions, reward range'''
    state_dim = env.observation_space.shape[0]
    if (len(env.observation_space.shape) > 1):
        state_dim = env.observation_space.shape
    return {
        'state_dim': state_dim,
        'state_bounds': np.transpose(
            [env.observation_space.low, env.observation_space.high]),
        'action_dim': env.action_space.n,
        'actions': list(range(env.action_space.n)),
        'reward_range': env.reward_range,
        'timestep_limit': env.spec.tags.get(
            'wrapper_config.TimeLimit.max_episode_steps')
    }


def timestamp():
    '''timestamp used for filename'''
    return '{:%Y-%m-%d_%H%M%S}'.format(datetime.now())


def timestamp_elapse(s1, s2):
    '''calculate the time elapsed between timestamps from s1 to s2'''
    FMT = '%Y-%m-%d_%H%M%S'
    delta_t = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    return str(delta_t)


def report_speed(real_time, total_t):
    '''Report on how fast each time step runs'''
    avg_speed = float(real_time)/float(total_t)
    logger.info('Mean speed: {:.4f} s/step'.format(avg_speed))


def format_obj_dict(obj, keys):
    if isinstance(obj, dict):
        return pp.pformat(
            {k: obj.get(k) for k in keys})
    else:
        return pp.pformat(
            {k: getattr(obj, k, None) for k in keys})


def debug_agent_info(agent):
    logger.debug(
        "Agent info: {}".format(
            format_obj_dict(agent, ['learning_rate', 'n_epoch'])))
    logger.debug(
        "Memory info: size: {}".format(agent.memory.size()))
    logger.debug(
        "Policy info: {}".format(
            format_obj_dict(agent.policy, ['e'])))


def update_history(agent,
                   sys_vars,
                   total_t,
                   total_rewards):
    '''
    update the hisory (list of total rewards)
    max len = REWARD_MEAN_LEN
    then report status
    '''

    sys_vars['total_r_history'].append(total_rewards)
    policy = agent.policy
    sys_vars['explore_history'].append(
        getattr(policy, 'e', 0) or getattr(policy, 'tau', 0))
    avg_len = sys_vars['REWARD_MEAN_LEN']
    # Calculating mean_reward over last 100 episodes
    # case away from np for json serializable (dumb python)
    mean_rewards = float(np.mean(sys_vars['total_r_history'][-avg_len:]))
    solved = (mean_rewards >= sys_vars['SOLVED_MEAN_REWARD'])
    sys_vars['mean_rewards'] = mean_rewards
    sys_vars['total_rewards'] = total_rewards
    sys_vars['solved'] = solved
    live_plot(sys_vars)

    logger.debug(
        "RL Sys info: {}".format(
            format_obj_dict(
                sys_vars, ['epi', 't', 'total_rewards', 'mean_rewards'])))
    logger.debug('{:->30}'.format(''))
    check_session_ends(sys_vars)
    return sys_vars


def check_session_ends(sys_vars):
    if (sys_vars['solved'] or
            (sys_vars['epi'] == sys_vars['MAX_EPISODES'] - 1)):
        logger.info('Problem solved? {}. At epi: {}. Params: {}'.format(
            sys_vars['solved'], sys_vars['epi'],
            pp.pformat(sys_vars['PARAM'])))
    if not sys_vars['RENDER']:
        return
    plt.savefig('./data/{}.png'.format(sys_vars['GYM_ENV_NAME']))


def init_plotter(sys_vars):
    param = sys_vars['PARAM']
    if not sys_vars['RENDER']:
        return
    # initialize the plotters
    fig = plt.figure(facecolor='white', figsize=(8, 9))

    ax1 = fig.add_subplot(311,
                          frame_on=False,
                          title="learning rate: {}, "
                          "gamma: {}\ntotal rewards per episode".format(
                              str(param.get('learning_rate')),
                              str(param.get('gamma'))),
                          ylabel='total rewards')
    p1, = ax1.plot([], [])
    plotters['total rewards'] = (ax1, p1)

    ax1e = ax1.twinx()
    ax1e.set_ylabel('(epsilon or tau)').set_color('r')
    ax1e.set_frame_on(False)
    p1e, = ax1e.plot([], [], 'r')
    plotters['e'] = (ax1e, p1e)

    ax2 = fig.add_subplot(312,
                          frame_on=False,
                          title='mean rewards over last 100 episodes',
                          ylabel='mean rewards')
    p2, = ax2.plot([], [], 'g')
    plotters['mean rewards'] = (ax2, p2)

    ax3 = fig.add_subplot(313,
                          frame_on=False,
                          title='loss over time, episode',
                          ylabel='loss')
    p3, = ax3.plot([], [])
    plotters['loss'] = (ax3, p3)

    plt.tight_layout()  # auto-fix spacing
    plt.ion()  # for live plot


def live_plot(sys_vars):
    '''do live plotting'''
    if not sys_vars['RENDER']:
        return
    ax1, p1 = plotters['total rewards']
    p1.set_ydata(np.append(p1.get_ydata(), sys_vars['total_r_history'][-1]))
    p1.set_xdata(np.arange(len(p1.get_ydata())))
    ax1.relim()
    ax1.autoscale_view(tight=True, scalex=True, scaley=True)

    ax1e, p1e = plotters['e']
    p1e.set_ydata(np.append(p1e.get_ydata(), sys_vars['explore_history'][-1]))
    p1e.set_xdata(np.arange(len(p1e.get_ydata())))
    ax1e.relim()
    ax1e.autoscale_view(tight=True, scalex=True, scaley=True)

    ax2, p2 = plotters['mean rewards']
    p2.set_ydata(np.append(p2.get_ydata(), sys_vars['mean_rewards']))
    p2.set_xdata(np.arange(len(p2.get_ydata())))
    ax2.relim()
    ax2.autoscale_view(tight=True, scalex=True, scaley=True)

    ax3, p3 = plotters['loss']
    p3.set_ydata(sys_vars['loss'])
    p3.set_xdata(np.arange(len(p3.get_ydata())))
    ax3.relim()
    ax3.autoscale_view(tight=True, scalex=True, scaley=True)

    plt.draw()
    plt.pause(0.01)


# convert a dict of param ranges into
# a list of cartesian products of param_range
# e.g. {'a': [1,2], 'b': [3]} into
# [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
def param_product(default_param, param_range):
    keys = param_range.keys()
    range_vals = param_range.values()
    param_grid = []
    for vals in itertools.product(*range_vals):
        param = copy.deepcopy(default_param)
        param.update(dict(zip(keys, vals)))
        param_grid.append(param)
    return param_grid


def stringify_param_value(value):
    return value.__name__ if isinstance(value, type) else value


def stringify_param(param):
    return {k: stringify_param_value(param[k]) for k in param}


# convert a dict of param ranges into
# a list parameter settings corresponding
# to a line search of the param range
# for each param
# All other parameters set to default vals
def param_line_search(default_param, param_range):
    keys = param_range.keys()
    param_list = []
    for key in keys:
        vals = param_range[key]
        for val in vals:
            param = copy.deepcopy(default_param)
            param[key] = val
            param_list.append(param)
    return param_list
