import argparse
import collections
import copy
import itertools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from keras import backend as K
from os import path, environ
from textwrap import wrap

PARALLEL_PROCESS_NUM = mp.cpu_count()

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
parser.add_argument("-s", "--sess",
                    help="specifies session to run, see experiment_specs.json",
                    action="store",
                    type=str,
                    nargs='?',
                    dest="experiment_name",
                    default="dev_dqn")
parser.add_argument("-t", "--times",
                    help="number of times session is run",
                    action="store",
                    nargs='?',
                    type=int,
                    dest="times",
                    default=1)
parser.add_argument("-e", "--evals",
                    help="number of max trials ran: hyperopt max_evals",
                    action="store",
                    nargs='?',
                    type=int,
                    dest="max_evals",
                    default=2)
parser.add_argument("-l", "--line_search",
                    help="to use line_search instead of param_product",
                    action="store_const",
                    dest="line_search",
                    const=False,
                    default=False)
parser.add_argument("-p", "--param_selection",
                    help="run parameter selection if present",
                    action="store_true",
                    dest="param_selection",
                    default=False)
parser.add_argument("-g", "--graph",
                    help="plot metrics graphs live",
                    action="store_true",
                    dest="plot_graph",
                    default=False)
parser.add_argument("-a", "--analyze",
                    help="only run analyze_data",
                    action="store_true",
                    dest="analyze_only",
                    default=False)
args = parser.parse_args([]) if environ.get('CI') else parser.parse_args()

# Goddam python logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.setLevel(args.loglevel)
logger.addHandler(handler)
logger.propagate = False


def log_self(subject):
    logger.info('{}, params: {}'.format(
        subject.__class__.__name__,
        to_json(subject.__dict__)))


def wrap_text(text):
    return '\n'.join(wrap(text, 60))


def print_line(line='-'):
    if environ.get('CI'):
        return
    _rows, columns = os.popen('stty size', 'r').read().split()
    line_str = line*int(columns)
    print(line_str)


def log_delimiter(msg, line='-'):
    print('{:\n>3}'.format(''))
    print_line(line)
    print(msg)
    print_line(line)
    print('{:\n>3}'.format(''))


def timestamp():
    '''timestamp used for filename'''
    return '{:%Y-%m-%d_%H%M%S}'.format(datetime.now())


def timestamp_elapse(s1, s2):
    '''calculate the time elapsed between timestamps from s1 to s2'''
    FMT = '%Y-%m-%d_%H%M%S'
    delta_t = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    return str(delta_t)


def timestamp_elapse_to_seconds(s1):
    a = datetime.strptime(s1, '%H:%M:%S')
    secs = timedelta(hours=a.hour, minutes=a.minute, seconds=a.second).seconds
    return secs


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
    elif isinstance(o, list) or isinstance(o, tuple):
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


# format object and its properties into printable dict
def format_obj_dict(obj, keys):
    if isinstance(obj, dict):
        return to_json(
            {k: obj.get(k) for k in keys if obj.get(k) is not None})
    else:
        return to_json(
            {k: getattr(obj, k, None) for k in keys
             if getattr(obj, k, None) is not None})


# cast dict to have flat values (int, float, str)
def flat_cast_dict(d):
    for k in d:
        v = d[k]
        if not isinstance(v, (int, float)):
            d[k] = str(v)
    return d


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_module(GREF, dot_path):
    # get module from globals() by string dot_path
    path_arr = dot_path.split('.')
    # base level from globals
    mod = GREF.get(path_arr.pop(0))
    for deeper_path in path_arr:
        mod = getattr(mod, deeper_path)
    return mod


# convert a dict of param ranges into
# a list parameter settings corresponding
# to a line search of the param range
# for each param
# All other parameters set to default vals
def param_line_search(experiment_spec):
    default_param = experiment_spec['param']
    param_range = experiment_spec['param_range']
    keys = param_range.keys()
    param_list = []
    for key in keys:
        vals = param_range[key]
        for val in vals:
            param = copy.deepcopy(default_param)
            param[key] = val
            param_list.append(param)
    return param_list


# convert a dict of param ranges into
# a list of cartesian products of param_range
# e.g. {'a': [1,2], 'b': [3]} into
# [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
def param_product(experiment_spec):
    default_param = experiment_spec['param']
    param_range = experiment_spec['param_range']
    keys = param_range.keys()
    range_vals = param_range.values()
    param_grid = []
    for vals in itertools.product(*range_vals):
        param = copy.deepcopy(default_param)
        param.update(dict(zip(keys, vals)))
        param_grid.append(param)
    return param_grid


# for param selection
def generate_experiment_spec_grid(experiment_spec, param_grid):
    experiment_spec_grid = [{
        'experiment_name': experiment_spec['experiment_name'],
        'problem': experiment_spec['problem'],
        'Agent': experiment_spec['Agent'],
        'Memory': experiment_spec['Memory'],
        'Policy': experiment_spec['Policy'],
        'PreProcessor': experiment_spec['PreProcessor'],
        'param': param,
    } for param in param_grid]
    return experiment_spec_grid


def experiment_id_from_trial_id(trial_id):
    str_arr = trial_id.split('_')
    if str_arr[-1].startswith('t'):
        str_arr.pop()
    return '_'.join(str_arr)


def load_data_from_trial_id(trial_id):
    trial_id = trial_id.split(
        '/').pop().split('.').pop(0)
    experiment_id = experiment_id_from_trial_id(trial_id)
    data_filename = './data/{}/{}.json'.format(experiment_id, trial_id)
    try:
        data = json.loads(open(data_filename).read())
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warn('Failed to read JSON from {}'.format(data_filename))
        data = None
    return data


def load_data_array_from_experiment_id(experiment_id):
    # to load all ./data files for a series of trials
    experiment_id = experiment_id_from_trial_id(experiment_id)
    data_path = './data/{}'.format(experiment_id)
    trial_id_array = [
        f for f in os.listdir(data_path)
        if (path.isfile(path.join(data_path, f)) and
            f.startswith(experiment_id) and
            f.endswith('.json'))
    ]
    return list(filter(None, [load_data_from_trial_id(trial_id)
                              for trial_id in trial_id_array]))


def save_experiment_data(data_df, trial_id):
    experiment_id = experiment_id_from_trial_id(trial_id)
    filename = './data/{0}/experiment_data_{0}.csv'.format(experiment_id)
    data_df.to_csv(filename, index=False)
    logger.info(
        'experiment data saved to {}'.format(filename))


def configure_gpu():
    '''detect GPU options and configure'''
    if K.backend() != 'tensorflow':
        # skip directly if is not tensorflow
        return
    real_parallel_process_num = 1 if mp.current_process(
    ).name == 'MainProcess' else PARALLEL_PROCESS_NUM
    tf = K.tf
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=1./float(real_parallel_process_num))
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)
    return sess
