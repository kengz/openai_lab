# The trial logic and analysis
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
EXPERIMENT_SPECS = json.loads(open(
    path.join(ASSET_PATH, 'experiment_specs.json')).read())
for k in EXPERIMENT_SPECS:
    EXPERIMENT_SPECS[k]['experiment_name'] = k

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
    The base unit of an Trial
    An Trial for a config on repeat for k time
    will run k Sessions, each with identical experiment_spec
    for a problem, Agent, Memory, Policy, param.
    Handles its own data, plots and saves its own graphs
    Serialized by the parent trial_id with its session_id
    '''

    def __init__(self, trial, session_num=0, num_of_sessions=1):
        self.trial = trial
        self.session_num = session_num
        self.num_of_sessions = num_of_sessions
        self.session_id = self.trial.trial_id + \
            '_s' + str(self.session_num)
        log_delimiter('Init Session #{} of {}:\n{}'.format(
            self.session_num, self.num_of_sessions, self.session_id))

        self.experiment_spec = trial.experiment_spec
        self.problem = self.experiment_spec['problem']
        self.Agent = get_module(GREF, self.experiment_spec['Agent'])
        self.Memory = get_module(GREF, self.experiment_spec['Memory'])
        self.Policy = get_module(GREF, self.experiment_spec['Policy'])
        self.PreProcessor = get_module(GREF, self.experiment_spec['PreProcessor'])
        self.param = self.experiment_spec['param']

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
            self.trial.experiment_id, self.session_id)
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
                logger.error('Error in trial, terminating '
                             'further session from {}'.format(self.session_id))
                traceback.print_exc(file=sys.stdout)
                break
            if sys_vars['solved']:
                break

        self.clear_session()
        sys_vars['time_end'] = timestamp()
        sys_vars['time_taken'] = timestamp_elapse(
            sys_vars['time_start'], sys_vars['time_end'])

        progress = 'Progress: Trial #{} Session #{} of {} done'.format(
            self.trial.trial_num,
            self.session_num, self.num_of_sessions)
        log_delimiter('End Session:\n{}\n{}'.format(
            self.session_id, progress))
        return sys_vars


class Trial(object):

    '''
    An Trial for a config on repeat for k time
    will run k Sessions, each with identical experiment_spec
    for a problem, Agent, Memory, Policy, param.
    Will spawn as many Sessions for repetition
    Handles all the data from sessions
    to provide an trial-level summary for a experiment_spec
    Its trial_id is serialized by
    problem, Agent, Memory, Policy and timestamp
    Data Requirements:
    JSON, single file, quick and useful summary,
    replottable data, rerunnable specs
    Keys:
    all below X array of hyper param selection:
    - trial_id
    - metrics
        - <metrics>
        - time_start
        - time_end
        - time_taken
    - experiment_spec (so we can plug in directly again to rerun)
    - stats
    - sys_vars_array
    '''

    def __init__(self, experiment_spec, times=1,
                 trial_num=0, num_of_trials=1,
                 run_timestamp=timestamp(),
                 experiment_id_override=None):

        self.experiment_spec = experiment_spec
        self.experiment_name = experiment_spec.get('experiment_name')
        param_range = EXPERIMENT_SPECS.get(self.experiment_name).get('param_range')
        self.param_variables = list(
            param_range.keys()) if param_range else []
        self.experiment_spec.pop('param_range', None)  # single exp, del range
        self.data = None
        self.times = times
        self.trial_num = trial_num
        self.num_of_trials = num_of_trials
        self.run_timestamp = run_timestamp
        self.experiment_id = experiment_id_override or '{}_{}_{}_{}_{}_{}'.format(
            experiment_spec['problem'],
            experiment_spec['Agent'].split('.').pop(),
            experiment_spec['Memory'].split('.').pop(),
            experiment_spec['Policy'].split('.').pop(),
            experiment_spec['PreProcessor'].split('.').pop(),
            self.run_timestamp
        )
        self.trial_id = self.experiment_id + '_t' + str(self.trial_num)
        self.base_dir = './data/{}'.format(self.experiment_id)
        os.makedirs(self.base_dir, exist_ok=True)
        self.base_filename = './data/{}/{}'.format(
            self.experiment_id, self.trial_id)
        self.data_filename = self.base_filename + '.json'
        log_delimiter('Init Trial #{} of {}:\n{}'.format(
            self.trial_num, self.num_of_trials,
            self.trial_id), '=')

    def save(self):
        '''save the entire trial data grid from inside run()'''
        with open(self.data_filename, 'w') as f:
            f.write(to_json(self.data))
        logger.info(
            'Session complete, data saved to {}'.format(self.data_filename))

    def to_stop(self):
        '''check of trial should be continued'''
        failed = self.data['stats']['solved_ratio_of_sessions'] == 0.
        if failed:
            logger.info(
                'Failed trial, terminating sessions for {}'.format(
                    self.trial_id))
        return failed

    def is_completed(self):
        '''check if the trial is already completed, if so dont run'''
        trial_data = load_data_from_trial_id(self.trial_id)
        if trial_data is None:
            # no data yet, confirmed incomplete
            return False
        else:
            stats = trial_data['stats']
            num_of_sessions = stats['num_of_sessions']
            solved_num_of_sessions = stats['solved_num_of_sessions']
            # hasn't ran enough times but is promising, is incomplete
            is_incomplete = (num_of_sessions < self.times and
                             solved_num_of_sessions == num_of_sessions)
            return not is_incomplete

    def run(self):
        '''
        helper: run a trial for Session
        a number of times times given a experiment_spec from gym_specs
        '''
        # TODO fix fresh run warning
        # if self.is_completed():
        #     return load_data_from_trial_id(self.trial_id)
        log_delimiter('Run Trial #{} of {}:\n{}'.format(
            self.trial_num, self.num_of_trials,
            self.trial_id), '=')
        configure_gpu()
        time_start = timestamp()
        sys_vars_array = []
        for s in range(self.times):
            sess = Session(trial=self,
                           session_num=s, num_of_sessions=self.times)
            sys_vars = sess.run()
            sys_vars_array.append(copy.copy(sys_vars))
            time_end = timestamp()
            time_taken = timestamp_elapse(time_start, time_end)

            self.data = {  # trial data
                'trial_id': self.trial_id,
                'metrics': {
                    # 'time_start': time_start,
                    # 'time_end': time_end,
                    'time_taken': time_taken,
                },
                'experiment_spec': self.experiment_spec,
                'stats': None,
                'sys_vars_array': sys_vars_array,
            }
            compose_data(self)
            self.save()  # progressive update, write every session completion

            if self.to_stop():
                break

        progress = 'Progress: Trial #{} of {} done'.format(
            self.trial_num, self.num_of_trials)
        log_delimiter(
            'End Trial:\n{}\n{}'.format(
                self.trial_id, progress), '=')
        return self.data


def analyze_experiment(trial_or_experiment_id):
    '''plot from a saved data by init sessions for each sys_vars'''
    experiment_id = experiment_id_from_trial_id(trial_or_experiment_id)
    experiment_data = load_data_array_from_experiment_id(experiment_id)
    return analyze_data(experiment_data)


def run(experiment_name_id_spec, times=1,
        param_selection=False,
        analyze_only=False, **kwargs):
    '''
    primary method:
    specify:
    - experiment_name(str) or experiment_spec(Dict): run new trial,
    - experiment_id(str): rerun any incomplete trials from the experiment
    - trial_id(str): rerun trial from data
    - trial_id(str) with analyze_only=True: plot graphs from data
    This runs all trials, specified by the obtained experiment_spec
    for a specified number of sessions per trial
    Multiple trials are ran if param_selection=True
    '''
    # run plots on data only
    if analyze_only:
        analyze_experiment(experiment_name_id_spec)
        return

    # set experiment_spec based on input
    if isinstance(experiment_name_id_spec, str):
        if len(experiment_name_id_spec.split('_')) >= 4:
            trial_data = load_data_from_trial_id(experiment_name_id_spec)
            experiment_spec = trial_data['experiment_spec']
        else:
            experiment_spec = EXPERIMENT_SPECS.get(experiment_name_id_spec)
    else:
        experiment_spec = experiment_name_id_spec

    # compose grid and run param selection
    if param_selection:
        hopt_kwargs = {
            'experiment_spec': experiment_spec,
            'times': times
        }
        hopt_kwargs.update(kwargs)
        hopt = BruteHyperOptimizer(Trial, **hopt_kwargs)
        # hopt = HyperoptHyperOptimizer(Trial, **hopt_kwargs)
        experiment_data = hopt.run()
    else:
        trial = Trial(experiment_spec, times=times)
        trial_data = trial.run()
        experiment_data = [trial_data]

    return analyze_data(experiment_data)
