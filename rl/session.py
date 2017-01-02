import gc
import gym
import json
import multiprocessing as mp
from datetime import datetime
from functools import partial
from rl.spec import game_specs
from rl.util import *


class Session(object):

    '''
    main.py calls this
    The base class for running a session of
    a DQN Agent, at a problem, with agent params
    '''

    def __init__(self, problem, Agent, Memory, Policy, param):
        self.problem = problem
        self.Agent = Agent
        self.Memory = Memory
        self.Policy = Policy
        self.param = param

    def run_episode(self, sys_vars, env, agent):
        '''run ane episode, return sys_vars'''
        state = env.reset()
        agent.memory.reset_state(state)
        total_rewards = 0
        debug_agent_info(agent)

        for t in range(agent.env_spec['timestep_limit']):
            sys_vars['t'] = t  # update sys_vars t
            if sys_vars.get('RENDER'):
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.add_exp(action, reward, next_state, done)
            agent.update(sys_vars)
            if agent.to_train(sys_vars):
                agent.train(sys_vars)

            state = next_state
            total_rewards += reward
            if done:
                break
        update_history(agent, sys_vars, t, total_rewards)
        return sys_vars

    def run(self):
        '''run a session of agent'''
        sys_vars = init_sys_vars(
            self.problem, self.param)  # rl system, see util.py
        env = gym.make(sys_vars['GYM_ENV_NAME'])
        agent = self.Agent(get_env_spec(env), **self.param)
        memory = self.Memory(**self.param)
        policy = self.Policy(**self.param)
        agent.compile(memory, policy)
        logger.info('Compiled Agent, Memory, Policy')

        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            self.run_episode(sys_vars, env, agent)
            if sys_vars['solved']:
                break

        gc.collect()  # manual gc to fix TF issue 3388
        return sys_vars


def experiment_analytics(data):
    '''
    helper: define the performance metric
    given data from an experiment
    '''
    sys_vars_array = data['sys_vars_array']
    mean_r_array = [sys_vars['mean_rewards'] for sys_vars in sys_vars_array]
    metrics = {
        'experiment_mean': np.mean(mean_r_array),
        'experiment_std': np.std(mean_r_array),
    }
    return metrics


def save_experiment_data(data_grid):
    '''
    log the entire experiment data grid from inside run()
    '''
    # sort data, best first
    data_grid.sort(
        key=lambda data: data['metrics']['experiment_mean'],
        reverse=True)
    timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.now())
    filename = './data/{}_{}_{}_{}_{}.json'.format(
        data_grid[0]['sess_spec']['problem'],
        data_grid[0]['sess_spec']['Agent'],
        data_grid[0]['sess_spec']['Memory'],
        data_grid[0]['sess_spec']['Policy'],
        timestamp
    )
    with open(filename, 'w') as f:
        json.dump(data_grid, f, indent=2, sort_keys=True)
    logger.info('Experiment complete, written to data/')


def run_single_exp(sess_spec, data_grid, times=1):
    '''
    helper: run a experiment for Session
    a number of times times given a sess_spec from gym_specs
    '''
    start_time = datetime.now().isoformat()
    sess = Session(problem=sess_spec['problem'],
                   Agent=sess_spec['Agent'],
                   Memory=sess_spec['Memory'],
                   Policy=sess_spec['Policy'],
                   param=sess_spec['param'])
    sys_vars_array = [sess.run() for i in range(times)]
    end_time = datetime.now().isoformat()
    data = {  # experiment data
        'start_time': start_time,
        'sess_spec': stringify_param(sess_spec),
        'sys_vars_array': sys_vars_array,
        'metrics': None,
        'end_time': end_time,
    }
    data.update({'metrics': experiment_analytics(data)})
    # progressive update of data_grid, write when an exp is done
    data_grid.append(data)
    save_experiment_data(data_grid)
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
