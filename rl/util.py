# everything shall start from 0
import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.handlers.pop()  # override the gym's handler


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


def check_sys_vars(sys_vars):
    return


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
