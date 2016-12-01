import gym
import multiprocessing as mp
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


def run_sess(sess_spec):
    '''
    helper: run a Session given a sess_spec from gym_specs
    '''
    sess = Session(problem=sess_spec['problem'],
                   Agent=sess_spec['Agent'],
                   Memory=sess_spec['Memory'],
                   Policy=sess_spec['Policy'],
                   param=sess_spec['param'])
    sys_vars = sess.run()
    return sys_vars


def run_sess_avg(sess_spec, times=1):
    '''
    helper: run a Session for the number of times specified,
    then average the 'mean_rewards'
    Used for hyper param selection
    '''
    param = sess_spec['param']
    logger.info(
        'Running session average with param = {}'.format(pp.pformat(
            param)))
    sess_mean_rewards = []
    # explicit loop necessary to circumvent TF bug
    # see https://github.com/tensorflow/tensorflow/issues/3388
    for i in range(times):
        sess_mean_rewards.append(run_sess(sess_spec)['mean_rewards'])
    sess_avg = np.mean(sess_mean_rewards)
    logger.info(
        'Session average mean_rewards: {} with param = {}'.format(
            sess_avg, pp.pformat(param)))
    return {'param': param, 'sess_avg_mean_rewards': sess_avg}


def run(sess_name):
    '''
    Wrapper for main.py to run session by name pointing to specs
    '''
    sess_spec = game_specs.get(sess_name)
    return run_sess(sess_spec)


def run_avg(sess_name):
    '''
    Like run(), but calls run_sess_avg() internally
    '''
    sess_spec = game_specs.get(sess_name)
    return run_sess_avg(sess_spec)


def run_param_selection(sess_name):
    '''
    Run hyper parameter selection with run_sess_avg
    draws from sess_spec['param_range'] to construct
    a param_grid, then sess_spec_grid,
    to run run_sess_avg on each
    '''
    sess_spec = game_specs.get(sess_name)
    param_grid = param_product(sess_spec['param'], sess_spec['param_range'])
    sess_spec_grid = [{
        'problem': sess_spec['problem'],
        'Agent': sess_spec['Agent'],
        'Memory': sess_spec['Memory'],
        'Policy': sess_spec['Policy'],
        'param': param,
    } for param in param_grid]

    p = mp.Pool(mp.cpu_count())
    avg_runs = list(p.map(run_sess_avg, sess_spec_grid))
    avg_runs.sort(key=lambda pm: pm['sess_avg_mean_rewards'], reverse=True)
    logger.info('Ranked params, from the best:'
                ''.format(pp.pformat(avg_runs)))
    return avg_runs
