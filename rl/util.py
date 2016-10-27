# everything shall start from 0
import pprint
import logging
import numpy as np
from collections import deque

pp = pprint.PrettyPrinter(indent=2)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.handlers.pop()  # override the gym's handler

required_sys_keys = {
    'RENDER',
    'GYM_ENV_NAME',
    'SOLVED_MEAN_REWARD',
    'MAX_STEPS',
    'MAX_EPISODES',
    'MAX_HISTORY',
    'epi',
    'history',
    'mean_rewards',
    'solved'
}


def check_sys_vars(sys_vars):
    '''ensure the requried RL system vars are specified'''
    sys_keys = sys_vars.keys()
    assert all(k in sys_keys for k in required_sys_keys)


def reset_sys_vars(sys_vars):
    '''reset and check RL system vars before each new session'''
    sys_vars['epi'] = 0
    sys_vars['history'] = deque(maxlen=sys_vars.get('MAX_HISTORY'))
    sys_vars['mean_rewards'] = 0
    sys_vars['solved'] = False
    check_sys_vars(sys_vars)
    return sys_vars


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
    max len = MAX_HISTORY
    then report status
    '''
    sys_vars['history'].append(total_rewards)
    mean_rewards = np.mean(sys_vars['history'])
    solved = (mean_rewards >= sys_vars['SOLVED_MEAN_REWARD'])
    sys_vars['mean_rewards'] = mean_rewards
    sys_vars['solved'] = solved
    logs = [
        '',
        'Episode: {}, total t: {}, total reward: {}'.format(
            sys_vars['epi'], total_t, total_rewards),
        'Mean rewards over last {} episodes: {:.4f}'.format(
            sys_vars['MAX_HISTORY'], mean_rewards),
        '{:->20}'.format(''),
    ]
    logger.info('\n'.join(logs))
    if solved or (sys_vars['epi'] == sys_vars['MAX_EPISODES'] - 1):
        logger.info('Problem solved? {}'.format(solved))
    return sys_vars
