import gym
import json
import multiprocessing as mp
from datetime import datetime
from functools import partial
from rl.agent import *
from rl.memory import *
from rl.policy import *
from rl.util import *

# Dict of specs runnable on a Session
game_specs = {
    'dummy': {
        'problem': 'CartPole-v0',
        'Agent': dummy.Dummy,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {}
    },
    'q_table': {
        'problem': 'CartPole-v0',
        'Agent': q_table.QTable,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'learning_rate': 0.01,
            'gamma': 0.99,
            'exploration_anneal_episodes': 200,
        }
    },
    'dqn': {
        'problem': 'CartPole-v0',
        'Agent': dqn.DQN,
        'Memory': LinearMemoryWithForgetting,
        'Policy': BoltzmannPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.02,
            'gamma': 0.99,
            'hidden_layers_shape': [4],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 10,
        },
        'param_range': {
            'learning_rate': [0.01, 0.05, 0.1],
            'gamma': [0.99],
            'exploration_anneal_episodes': [50, 100],
        }
    },
    'double_dqn': {
        'problem': 'CartPole-v0',
        'Agent': double_dqn.DoubleDQN,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.01,
            'batch_size': 32,
            'gamma': 0.99,
            'hidden_layers_shape': [4],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 180,
        }
    },
    'mountain_double_dqn': {
        'problem': 'MountainCar-v0',
        'Agent': mountain_double_dqn.MountainDoubleDQN,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.01,
            'batch_size': 128,
            'gamma': 0.99,
            'hidden_layers_shape': [8, 8],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 300,
        }
    },
    'lunar_dqn': {
        'problem': 'LunarLander-v2',
        'Agent': lunar_dqn.LunarDQN,
        'Memory': LinearMemoryWithForgetting,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'gamma': 0.98,
            'hidden_layers_shape': [200, 100],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 300,
        }
    },
    'lunar_double_dqn': {
        'problem': 'LunarLander-v2',
        'Agent': lunar_double_dqn.LunarDoubleDQN,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.01,
            'batch_size': 64,
            'gamma': 0.99,
            'hidden_layers_shape': [200, 100],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 500,
        }
    }
}


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
        logger.debug(
            "DQN Agent param: {} Mem size: {}".format(
                pp.pformat(
                    {k: getattr(agent, k, None)
                     for k in ['e', 'learning_rate', 'batch_size', 'n_epoch']}
                ), agent.memory.size()))

        for t in range(env.spec.timestep_limit):
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
    filename = '{}_data_grid.json'.format(datetime.now().date().isoformat())
    # TODO WHAT THE FUCK IS THIS PYTHON CANT SERIALIZE BOOLEAN 'False'?
    del data_grid[0]['sys_vars_array'][0]['solved']
    with open(filename, 'w') as f:
        f.write(json.dumps(data_grid))
    logger.info('Experiments complete, data written to data_grid.json')


def run_single_exp(sess_spec, times=1):
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
    data.update({
        'metrics': experiment_analytics(data),
    })
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
    else:
        sess_spec_grid = [sess_spec]

    if run_param_selection or times > 1:
        p = mp.Pool(mp.cpu_count())
        data_grid = list(p.map(
            partial(run_single_exp, times=times),
            sess_spec_grid))
    else:
        data_grid = list(map(
            partial(run_single_exp, times=times),
            sess_spec_grid))

    save_experiment_data(data_grid)
    return data_grid

# TODO sort json by ordereddict
# TODO change to progressively write data per exp done, save from all-fail
