import gc
import gym
import json
import multiprocessing as mp
from datetime import datetime
from functools import partial
from rl.spec import game_specs
from rl.util import *
from rl.state_preprocessing import *


class Session(object):

    '''
    main.py calls this
    The base class for running a session of
    a DQN Agent, at a problem, with agent params
    '''

    def __init__(self, problem, Agent, Memory, Policy, param, sess_name):
        self.problem = problem
        self.Agent = Agent
        self.Memory = Memory
        self.Policy = Policy
        self.param = param
        self.sess_name = sess_name

    def run_episode(self, sys_vars, env, agent):
        '''run ane episode, return sys_vars'''
        state = env.reset()
        print("State shape: {}".format(state.shape))
        agent.memory.reset_state(state)
        total_rewards = 0
        debug_agent_info(agent)

        # Dummy previous states
        # print(state.shape)
        # print(state.shape[0])
        # exit(0)
        previous_state = np.zeros(state.shape)
        pre_previous_state = np.zeros(state.shape)
        pre_pre_previous_state = np.zeros(state.shape)
        if (previous_state.ndim==1):
            previous_state = np.zeros([state.shape[0]])
            pre_previous_state = np.zeros([state.shape[0]])
            pre_pre_previous_state = np.zeros([state.shape[0]])

        # Temp memory buffer to hold last n experiences
        # Enables data preprocessing, inc. diff and stack
        temp_exp_mem = []
        for t in range(agent.env_spec['timestep_limit']):
            sys_vars['t'] = t  # update sys_vars t
            if sys_vars.get('RENDER'):
                env.render()

            # State preprocessing for action selection, determined by spec.py
            proc_state = state # Default, no processing
            if 'state_preprocessing' in game_specs[self.sess_name]['param']:
                if (game_specs[self.sess_name]['param']['state_preprocessing'] == 'diff'):
                    proc_state = action_sel_processing_diff_states(state, previous_state)
                elif (game_specs[self.sess_name]['param']['state_preprocessing'] == 'concat'):
                    proc_state = action_sel_processing_stack_states(state, previous_state)
                elif (game_specs[self.sess_name]['param']['state_preprocessing'] == 'atari'):
                    proc_state = action_sel_processing_atari_states(state, previous_state,
                                                                        pre_previous_state,
                                                                        pre_pre_previous_state)
            if (t == 0):
                print("Initial state dim: {}".format(proc_state.shape))

            action = agent.select_action(proc_state)
            next_state, reward, done, info = env.step(action)
            temp_exp_mem.append([state, action, reward, next_state, done])
            # Buffer currently set to hold only last 4 experiences
            # Amount needed for Atari games preprocessing
            if (len(temp_exp_mem) > 4):
                del temp_exp_mem[0]   

            # State preprocessing for memory, determined by spec.py
            if 'state_preprocessing' in game_specs[self.sess_name]['param']:
                if (game_specs[self.sess_name]['param']['state_preprocessing'] == 'diff'):
                    run_state_processing_diff_states(agent, temp_exp_mem, t)
                elif (game_specs[self.sess_name]['param']['state_preprocessing'] == 'concat'):
                    run_state_processing_stack_states(agent, temp_exp_mem, t)
                elif (game_specs[self.sess_name]['param']['state_preprocessing'] == 'atari'):
                    run_state_processing_atari(agent, temp_exp_mem, t)
                else:
                    # Default: no processing
                    run_state_processing_none(agent, temp_exp_mem, t)    
            else:
                run_state_processing_none(agent, temp_exp_mem, t)
            agent.update(sys_vars)

            # t comparison indicates when to start training
            # Ideally should start at 3 so can be used for all tasks
            # E.g. If taking the difference of current and previous state can't train until t=1
            if (t >= 3) and agent.to_train(sys_vars):
                agent.train(sys_vars)
            
            pre_pre_previous_state = pre_previous_state
            pre_previous_state = previous_state
            previous_state = state
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
        
        ''' 
        Change state dim in env spec based on type of preprocessing
        '''
        env_spec = get_env_spec(env)
        if 'state_preprocessing' in game_specs[self.sess_name]['param']:
            if (game_specs[self.sess_name]['param']['state_preprocessing'] == 'concat'):
                env_spec['state_dim'] = env_spec['state_dim'] * 2
            elif (game_specs[self.sess_name]['param']['state_preprocessing'] == 'atari'):
                env_spec['state_dim'] = (84, 84, 4)

        agent = self.Agent(env_spec, **self.param)
        memory = self.Memory(**self.param)
        policy = self.Policy(**self.param)
        agent.compile(memory, policy)
        logger.info('Compiled Agent, Memory, Policy')

        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            self.run_episode(sys_vars, env, agent)
            if 'epi_change_learning_rate' in self.param and epi == self.param['epi_change_learning_rate']:
                agent.recompile_model(self.param['learning_rate'] / 10.0)
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
    mean_r_array = [sys_vars['mean_rewards'] for gsys_vars in sys_vars_array]
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


def run_single_exp(sess_spec, data_grid, sess_name, times=1):
    '''
    helper: run a experiment for Session
    a number of times times given a sess_spec from gym_specs
    '''
    start_time = datetime.now().isoformat()
    sess = Session(problem=sess_spec['problem'],
                   Agent=sess_spec['Agent'],
                   Memory=sess_spec['Memory'],
                   Policy=sess_spec['Policy'],
                   param=sess_spec['param'],
                   sess_name=sess_name)
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


def run(sess_name, run_param_selection=False, times=1, line_search=True):
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
        if line_search:
            param_grid = param_line_search(
            sess_spec['param'], sess_spec['param_range'])
        else:
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
            partial(run_single_exp, data_grid=data_grid, sess_name=sess_name, times=times),
            sess_spec_grid))
    else:
        run_single_exp(sess_spec, data_grid=data_grid, sess_name=sess_name, times=times)

    return data_grid
