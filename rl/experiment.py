# The experiment logic and analysis
RAND_SEED = 42
import numpy as np
np.random.seed(RAND_SEED)
np.seterr(all='raise')
from keras import backend as K
if K.backend() == 'tensorflow':
    K.tf.set_random_seed(RAND_SEED)
else:
    K.theano.tensor.shared_randomstreams.RandomStreams(seed=RAND_SEED)
import copy
import gym
import json
import traceback
from os import path, environ
from rl.util import *
from rl.agent import *
from rl.analytics import *
from rl.hyperoptimizer import *
from rl.memory import *
from rl.policy import *
from rl.preprocessor import *


GREF = globals()
ASSET_PATH = path.join(path.dirname(__file__), 'asset')
PROBLEMS = json.loads(open(
    path.join(ASSET_PATH, 'problems.json')).read())
SESS_SPECS = json.loads(open(
    path.join(ASSET_PATH, 'sess_specs.json')).read())
for k in SESS_SPECS:
    SESS_SPECS[k]['sess_name'] = k

# the keys and their defaults need to be implemented by a sys_var
# the constants (capitalized) are problem configs,
# set in asset/problems.json
REQUIRED_SYS_KEYS = {
    'RENDER': None,
    'GYM_ENV_NAME': None,
    'SOLVED_MEAN_REWARD': None,
    'MAX_EPISODES': None,
    'REWARD_MEAN_LEN': None,
    'epi': 0,
    't': 0,
    'done': False,
    'loss': [],
    'total_rewards_history': [],
    'explore_history': [],
    'mean_rewards_history': [],
    'mean_rewards': 0,
    'total_rewards': 0,
    'solved': False,
}


class Session(object):

    '''
    The base unit of an Experiment
    An Experiment for a config on repeat for k time
    will run k Sessions, each with identical sess_spec
    for a problem, Agent, Memory, Policy, param.
    Handles its own data, plots and saves its own graphs
    Serialized by the parent experiment_id with its session_id
    '''

    def __init__(self, experiment, session_num=0, num_of_sessions=1):
        self.experiment = experiment
        self.session_num = session_num
        self.num_of_sessions = num_of_sessions
        self.session_id = self.experiment.experiment_id + \
            '_s' + str(self.session_num)
        log_delimiter('Init Session #{} of {}:\n{}'.format(
            self.session_num, self.num_of_sessions, self.session_id))

        self.sess_spec = experiment.sess_spec
        self.problem = self.sess_spec['problem']
        self.Agent = get_module(GREF, self.sess_spec['Agent'])
        self.Memory = get_module(GREF, self.sess_spec['Memory'])
        self.Policy = get_module(GREF, self.sess_spec['Policy'])
        self.PreProcessor = get_module(GREF, self.sess_spec['PreProcessor'])
        self.param = self.sess_spec['param']

        # init all things, so a session can only be ran once
        self.sys_vars = self.init_sys_vars()
        self.env = gym.make(self.sys_vars['GYM_ENV_NAME'])
        self.preprocessor = self.PreProcessor(**self.param)
        self.env_spec = self.set_env_spec()
        self.agent = self.Agent(self.env_spec, **self.param)
        self.memory = self.Memory(**self.param)
        self.policy = self.Policy(**self.param)
        self.agent.compile(self.memory, self.policy, self.preprocessor)

        # data file and graph
        self.base_filename = './data/{}/{}'.format(
            self.experiment.prefix_id, self.session_id)
        self.graph_filename = self.base_filename + '.png'

        # for plotting
        self.grapher = Grapher(self)

    def init_sys_vars(self):
        '''
        init the sys vars for a problem by reading from
        asset/problems.json, then reset the other sys vars
        on reset will add vars (lower cases, see REQUIRED_SYS_KEYS)
        '''
        sys_vars = PROBLEMS[self.problem]
        if not args.render:
            sys_vars['RENDER'] = False
        if environ.get('CI'):
            sys_vars['RENDER'] = False
            if self.problem != 'DevCartPole-v0':
                sys_vars['MAX_EPISODES'] = 4
        self.sys_vars = sys_vars
        self.reset_sys_vars()
        return self.sys_vars

    def reset_sys_vars(self):
        '''reset and check RL system vars (lower case)
        before each new session'''
        for k in REQUIRED_SYS_KEYS:
            if k.islower():
                self.sys_vars[k] = copy.copy(REQUIRED_SYS_KEYS.get(k))
        self.check_sys_vars()
        return self.sys_vars

    def check_sys_vars(self):
        '''ensure the requried RL system vars are specified'''
        sys_keys = self.sys_vars.keys()
        assert all(k in sys_keys for k in REQUIRED_SYS_KEYS)

    def set_env_spec(self):
        '''Helper: return the env specs: dims, actions, reward range'''
        env = self.env
        state_dim = env.observation_space.shape[0]
        if (len(env.observation_space.shape) > 1):
            state_dim = env.observation_space.shape
        env_spec = {
            'state_dim': state_dim,
            'state_bounds': np.transpose(
                [env.observation_space.low, env.observation_space.high]),
            'action_dim': env.action_space.n,
            'actions': list(range(env.action_space.n)),
            'reward_range': env.reward_range,
            'timestep_limit': env.spec.tags.get(
                'wrapper_config.TimeLimit.max_episode_steps')
        }
        self.env_spec = self.preprocessor.preprocess_env_spec(
            env_spec)  # preprocess
        return self.env_spec

    def debug_agent_info(self):
        logger.debug(
            "Agent info: {}".format(
                format_obj_dict(
                    self.agent,
                    ['learning_rate', 'n_epoch'])))
        logger.debug(
            "Memory info: size: {}".format(self.agent.memory.size()))
        logger.debug(
            "Policy info: {}".format(
                format_obj_dict(self.agent.policy, ['e', 'tau'])))
        logger.debug(
            "PreProcessor info: {}".format(
                format_obj_dict(self.agent.preprocessor, [])))

    def check_end(self):
        '''check if session ends (if is last episode)
        do ending steps'''
        sys_vars = self.sys_vars

        logger.debug(
            "RL Sys info: {}".format(
                format_obj_dict(
                    sys_vars, ['epi', 't', 'total_rewards', 'mean_rewards'])))
        logger.debug('{:->30}'.format(''))

        if (sys_vars['solved'] or
                (sys_vars['epi'] == sys_vars['MAX_EPISODES'] - 1)):
            logger.info(
                'Problem solved? {}\nAt episode: {}\nParams: {}'.format(
                    sys_vars['solved'], sys_vars['epi'],
                    to_json(self.param)))
            self.env.close()

    def update_history(self):
        '''
        update the data per episode end
        '''

        sys_vars = self.sys_vars
        sys_vars['total_rewards_history'].append(sys_vars['total_rewards'])
        sys_vars['explore_history'].append(
            getattr(self.policy, 'e', 0) or getattr(self.policy, 'tau', 0))
        avg_len = sys_vars['REWARD_MEAN_LEN']
        # Calculating mean_reward over last 100 episodes
        # case away from np for json serializable (dumb python)
        mean_rewards = float(
            np.mean(sys_vars['total_rewards_history'][-avg_len:]))
        solved = (mean_rewards >= sys_vars['SOLVED_MEAN_REWARD'])
        sys_vars['mean_rewards'] = mean_rewards
        sys_vars['mean_rewards_history'].append(mean_rewards)
        sys_vars['solved'] = solved

        self.grapher.plot()
        self.check_end()
        return sys_vars

    def run_episode(self):
        '''run ane episode, return sys_vars'''
        sys_vars, env, agent = self.sys_vars, self.env, self.agent
        sys_vars['total_rewards'] = 0
        state = env.reset()
        processed_state = agent.preprocessor.reset_state(state)
        agent.memory.reset_state(processed_state)
        self.debug_agent_info()

        for t in range(agent.env_spec['timestep_limit']):
            sys_vars['t'] = t  # update sys_vars t
            if sys_vars.get('RENDER'):
                env.render()

            processed_state = agent.preprocessor.preprocess_state()
            action = agent.select_action(processed_state)
            next_state, reward, done, _info = env.step(action)
            processed_exp = agent.preprocessor.preprocess_memory(
                action, reward, next_state, done)
            if processed_exp is not None:
                agent.memory.add_exp(*processed_exp)

            sys_vars['done'] = done
            agent.update(sys_vars)
            if agent.to_train(sys_vars):
                agent.train(sys_vars)
            sys_vars['total_rewards'] += reward
            if done:
                break
        self.update_history()
        return sys_vars

    def clear_session(self):
        if K.backend() == 'tensorflow':
            K.clear_session()  # manual gc to fix TF issue 3388

    def run(self):
        '''run a session of agent'''
        log_delimiter('Run Session #{} of {}\n{}'.format(
            self.session_num, self.num_of_sessions, self.session_id))
        sys_vars = self.sys_vars
        sys_vars['time_start'] = timestamp()
        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            try:
                self.run_episode()
            except Exception:
                logger.error('Error in experiment, terminating '
                             'further session from {}'.format(self.session_id))
                traceback.print_exc(file=sys.stdout)
                break
            if sys_vars['solved']:
                break

        self.clear_session()
        sys_vars['time_end'] = timestamp()
        sys_vars['time_taken'] = timestamp_elapse(
            sys_vars['time_start'], sys_vars['time_end'])

        progress = 'Progress: Experiment #{} Session #{} of {} done'.format(
            self.experiment.experiment_num,
            self.session_num, self.num_of_sessions)
        log_delimiter('End Session:\n{}\n{}'.format(
            self.session_id, progress))
        return sys_vars


class Experiment(object):

    '''
    An Experiment for a config on repeat for k time
    will run k Sessions, each with identical sess_spec
    for a problem, Agent, Memory, Policy, param.
    Will spawn as many Sessions for repetition
    Handles all the data from sessions
    to provide an experiment-level summary for a sess_spec
    Its experiment_id is serialized by
    problem, Agent, Memory, Policy and timestamp
    Data Requirements:
    JSON, single file, quick and useful summary,
    replottable data, rerunnable specs
    Keys:
    all below X array of hyper param selection:
    - experiment_id
    - metrics
        - <metrics>
        - time_start
        - time_end
        - time_taken
    - sess_spec (so we can plug in directly again to rerun)
    - stats
    - sys_vars_array
    '''

    def __init__(self, sess_spec, times=1,
                 experiment_num=0, num_of_experiments=1,
                 run_timestamp=timestamp(),
                 prefix_id_override=None):

        self.sess_spec = sess_spec
        self.sess_name = sess_spec.get('sess_name')
        param_range = SESS_SPECS.get(self.sess_name).get('param_range')
        self.param_variables = list(
            param_range.keys()) if param_range else None
        self.sess_spec.pop('param_range', None)  # single exp, del range
        self.data = None
        self.times = times
        self.experiment_num = experiment_num
        self.num_of_experiments = num_of_experiments
        self.run_timestamp = run_timestamp
        self.prefix_id = prefix_id_override or '{}_{}_{}_{}_{}_{}'.format(
            sess_spec['problem'],
            sess_spec['Agent'].split('.').pop(),
            sess_spec['Memory'].split('.').pop(),
            sess_spec['Policy'].split('.').pop(),
            sess_spec['PreProcessor'].split('.').pop(),
            self.run_timestamp
        )
        self.experiment_id = self.prefix_id + '_e' + str(self.experiment_num)
        self.base_dir = './data/{}'.format(self.prefix_id)
        os.makedirs(self.base_dir, exist_ok=True)
        self.base_filename = './data/{}/{}'.format(
            self.prefix_id, self.experiment_id)
        self.data_filename = self.base_filename + '.json'
        log_delimiter('Init Experiment #{} of {}:\n{}'.format(
            self.experiment_num, self.num_of_experiments,
            self.experiment_id), '=')

    def save(self):
        '''save the entire experiment data grid from inside run()'''
        with open(self.data_filename, 'w') as f:
            f.write(to_json(self.data))
        logger.info(
            'Session complete, data saved to {}'.format(self.data_filename))

    def to_stop(self):
        '''check of experiment should be continued'''
        failed = self.data['stats']['solved_ratio_of_sessions'] < 1.
        if failed:
            logger.info(
                'Failed experiment, terminating sessions for {}'.format(
                    self.experiment_id))
        return failed

    def run(self):
        '''
        helper: run a experiment for Session
        a number of times times given a sess_spec from gym_specs
        '''
        configure_gpu()
        time_start = timestamp()
        sys_vars_array = []
        for s in range(self.times):
            sess = Session(experiment=self,
                           session_num=s, num_of_sessions=self.times)
            sys_vars = sess.run()
            sys_vars_array.append(copy.copy(sys_vars))
            time_end = timestamp()
            time_taken = timestamp_elapse(time_start, time_end)

            self.data = {  # experiment data
                'experiment_id': self.experiment_id,
                'metrics': {
                    # 'time_start': time_start,
                    # 'time_end': time_end,
                    'time_taken': time_taken,
                },
                'sess_spec': self.sess_spec,
                'stats': None,
                'sys_vars_array': sys_vars_array,
            }
            compose_data(self)
            self.save()  # progressive update, write every session completion

            if self.to_stop():
                break

        progress = 'Progress: Experiment #{} of {} done'.format(
            self.experiment_num, self.num_of_experiments)
        log_delimiter(
            'End Experiment:\n{}\n{}'.format(
                self.experiment_id, progress), '=')
        return self.data


def plot_experiment(experiment_or_prefix_id):
    '''plot from a saved data by init sessions for each sys_vars'''
    prefix_id = prefix_id_from_experiment_id(experiment_or_prefix_id)
    experiment_data_array = load_data_array_from_prefix_id(prefix_id)
    for data in experiment_data_array:
        sess_spec = data['sess_spec']
        experiment = Experiment(sess_spec, times=1,
                                prefix_id_override=prefix_id)
        # save with the right serialized filename
        experiment.experiment_id = data['experiment_id']
        num_of_sessions = len(data['sys_vars_array'])

        for s in range(num_of_sessions):
            sess = Session(experiment=experiment,
                           session_num=s, num_of_sessions=num_of_sessions)
            sys_vars = data['sys_vars_array'][s]
            sess.sys_vars = sys_vars
            sess.grapher.plot()
            sess.clear_session()


def run(sess_name_id_spec, times=1,
        param_selection=False,
        plot_only=False, **kwargs):
    '''
    primary method:
    specify:
    - sess_name(str) or sess_spec(Dict): run new experiment,
    - experiment_id(str): rerun experiment from data
    - experiment_id(str) with plot_only=True: plot graphs from data
    This runs all experiments, specified by the obtained sess_spec
    for a specified number of sessions per experiment
    Multiple experiments are ran if param_selection=True
    '''
    # run plots on data only
    if plot_only:
        plot_experiment(sess_name_id_spec)
        return

    # set sess_spec based on input
    if isinstance(sess_name_id_spec, str):
        if len(sess_name_id_spec.split('_')) >= 4:
            data = load_data_from_experiment_id(sess_name_id_spec)
            sess_spec = data['sess_spec']
        else:
            sess_spec = SESS_SPECS.get(sess_name_id_spec)
    else:
        sess_spec = sess_name_id_spec

    # compose grid and run param selection
    if param_selection:
        hopt_kwargs = {
            'sess_spec': sess_spec,
            'times': times
        }
        hopt_kwargs.update(kwargs)
        hopt = BruteHyperOptimizer(Experiment, **hopt_kwargs)
        # hopt = HyperoptHyperOptimizer(Experiment, **hopt_kwargs)
        experiment_data_array = hopt.run()
    else:
        experiment = Experiment(sess_spec, times=times)
        experiment_data = experiment.run()
        experiment_data_array = [experiment_data]

    return analyze_param_space(experiment_data_array)
