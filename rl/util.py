import argparse
import copy
import itertools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import pprint
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

# TODO absorb into session too

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


# own custom sorted json serializer, cuz python
def to_json(o, level=0):
    INDENT = 2
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k in sorted(o.keys()):
            v = o[k]
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level+1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + \
            ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError(
            "Unknown type '%s' for json serialization" % str(type(o)))
    return ret


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
