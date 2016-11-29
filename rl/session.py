import gym
import multiprocessing as mp
from rl.util import *
from rl.memory import LinearMemory, LeftTailMemory, LinearMemoryWithForgetting
from rl.agent import *

# Dict of specs runnable on a Session
game_specs = {
    'dummy': {
        'Agent': dummy.Dummy,
        'problem': 'CartPole-v0',
        'Num_experiences': 1,
        'Memory': LinearMemory,
        'param': {}
    },
    'q_table': {
        'Agent': q_table.QTable,
        'problem': 'CartPole-v0',
        'Num_experiences': 1,
        'Memory': LinearMemory,
        'param': {'e_anneal_episodes': 200,
                  'learning_rate': 0.01,
                  'gamma': 0.99}
    },
    'dqn': {
        'Agent': dqn.DQN,
        'problem': 'CartPole-v0',
        'Num_experiences': 1,
        'Memory': LinearMemoryWithForgetting,
        'param': {'e_anneal_episodes': 50,
                  'learning_rate': 0.01,
                  'gamma': 0.99,
                  'hidden_layers_shape': [4],
                  'hidden_layers_activation': 'sigmoid'},
        'param_range': {
            'e_anneal_episodes': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'gamma': [0.99]
        }
    },
    'double_dqn': {
        'Agent': double_dqn.DoubleDQN,
        'problem': 'CartPole-v0',
        'Num_experiences': 1,
        'Memory': LinearMemory,
        'param': {'e_anneal_episodes': 180,
                  'learning_rate': 0.01,
                  'batch_size': 32,
                  'gamma': 0.99,
                  'hidden_layers_shape': [4],
                  'hidden_layers_activation': 'sigmoid'}
    },
    'mountain_double_dqn': {
        'Agent': mountain_double_dqn.MountainDoubleDQN,
        'problem': 'MountainCar-v0',
        'Num_experiences': 1,
        'Memory': LinearMemory,
        'param': {'e_anneal_episodes': 300,
                  'learning_rate': 0.01,
                  'batch_size': 128,
                  'gamma': 0.99,
                  'hidden_layers_shape': [8, 8],
                  'hidden_layers_activation': 'sigmoid'}
    },
    'lunar_dqn': {
        'Agent': lunar_dqn.LunarDQN,
        'problem': 'LunarLander-v2',
        'Num_experiences': 4,
        'Memory': LinearMemoryWithForgetting,
        'param': {'e_anneal_episodes': 300,
                  'learning_rate': 0.001,
                  'batch_size': 32,
                  'gamma': 0.98,
                  'hidden_layers_shape': [200, 100],
                  'hidden_layers_activation': 'relu'}
    },
    'lunar_double_dqn': {
        'Agent': lunar_double_dqn.LunarDoubleDQN,
        'problem': 'LunarLander-v2',
        'Num_experiences': 1,
        'Memory': LinearMemory,
        'param': {'e_anneal_episodes': 500,
                  'learning_rate': 0.01,
                  'batch_size': 64,
                  'gamma': 0.99,
                  'hidden_layers_shape': [200, 100],
                  'hidden_layers_activation': 'relu'}
    }
}


class Session(object):

    '''
    main.py calls this
    The base class for running a session of
    a DQN Agent, at a problem, with agent params
    '''

    def __init__(self, Agent, problem, num_experiences, memory, param):
        self.Agent = Agent
        self.problem = problem
        self.param = param
        self.num_experiences = num_experiences
        self.memory = memory

    def run_episode(self, sys_vars, env, agent, replay_memory):
        '''run ane episode, return sys_vars'''
        state = env.reset()
        replay_memory.reset_state(state)
        total_rewards = 0
        logger.debug(
            "DQN Agent param: {} Num experiences: {} Mem size: {}".format(
                pp.pformat(
                    {k: getattr(agent, k, None)
                     for k in ['e', 'learning_rate', 'batch_size', 'n_epoch']}
                ), self.num_experiences, len(replay_memory.exp['states'])))

        for t in range(env.spec.timestep_limit):
            sys_vars['t'] = t  # update sys_vars t
            if sys_vars.get('RENDER'):
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_memory.add_exp(action, reward, next_state, done)
            agent.update(sys_vars, replay_memory)
            # Get n experiences before training model
            to_train = (
                (t != 0 and t % self.num_experiences == 0) or
                t == (env.spec.timestep_limit-1) or
                done)
            if to_train:
                agent.train(sys_vars, replay_memory)
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
        env_spec = get_env_spec(env)
        replay_memory = self.memory(env_spec)
        agent = self.Agent(env_spec, **self.param)

        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            self.run_episode(sys_vars, env, agent, replay_memory)
            if sys_vars['solved']:
                break

        return sys_vars


def run_sess(sess_spec):
    '''
    helper: run a Session given a sess_spec from gym_specs
    '''
    sess = Session(sess_spec['Agent'],
                   problem=sess_spec['problem'],
                   num_experiences=sess_spec['Num_experiences'],
                   memory=sess_spec['Memory'],
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
    param_range = sess_spec['param_range']
    param_grid = param_product(param_range)
    sess_spec_grid = [{
        'Agent': sess_spec['Agent'],
        'problem': sess_spec['problem'],
        'Num_experiences': sess_spec['Num_experiences'],
        'Memory': sess_spec['Memory'],
        'param': param
    } for param in param_grid]

    p = mp.Pool(mp.cpu_count())
    avg_runs = list(p.map(run_sess_avg, sess_spec_grid))
    avg_runs.sort(key=lambda pm: pm['sess_avg_mean_rewards'], reverse=True)
    logger.info('Ranked params, from the best:'
                ''.format(pp.pformat(avg_runs)))
    return avg_runs
