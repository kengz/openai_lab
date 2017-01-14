# The experiment logic and analysis
import copy
import gym
import json
import os
import matplotlib
matplotlib.rcParams['backend'] = 'agg' if os.environ.get('CI') else 'TkAgg'
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from functools import partial
from keras import backend as K
from rl.spec import game_specs
from rl.util import *

plt.rcParams['toolbar'] = 'None'  # mute matplotlib toolbar

PROBLEMS = json.loads(open(
    path.join(path.dirname(__file__), 'asset', 'problems.json')).read())

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
    'loss': [],
    'total_r_history': [],
    'explore_history': [],
    'mean_rewards': 0,
    'total_rewards': 0,
    'solved': False,
}

# Session respawn on taking json data can:
# - use sess_spec to rerun completely
# - use data to just replot


class Grapher(object):

    '''
    Grapher object that belongs to a Session
    to draw graphs from its data
    '''

    def __init__(self, session):
        self.session = session
        self.subgraphs = {}
        self.figure = plt.figure(facecolor='white', figsize=(8, 9))
        self.init_figure()

    def init_figure(self):
        if not self.session.sys_vars['RENDER']:
            return
        # graph 1
        ax1 = self.figure.add_subplot(
            311,
            frame_on=False,
            title="learning rate: {}, "
            "gamma: {}\ntotal rewards per episode".format(
                str(self.session.param.get('learning_rate')),
                str(self.session.param.get('gamma'))),
            ylabel='total rewards')
        p1, = ax1.plot([], [])
        self.subgraphs['total rewards'] = (ax1, p1)

        ax1e = ax1.twinx()
        ax1e.set_ylabel('(epsilon or tau)').set_color('r')
        ax1e.set_frame_on(False)
        p1e, = ax1e.plot([], [], 'r')
        self.subgraphs['e'] = (ax1e, p1e)

        # graph 2
        ax2 = self.figure.add_subplot(
            312,
            frame_on=False,
            title='mean rewards over last 100 episodes',
            ylabel='mean rewards')
        p2, = ax2.plot([], [], 'g')
        self.subgraphs['mean rewards'] = (ax2, p2)

        # graph 3
        ax3 = self.figure.add_subplot(
            313,
            frame_on=False,
            title='loss over time, episode',
            ylabel='loss')
        p3, = ax3.plot([], [])
        self.subgraphs['loss'] = (ax3, p3)

        plt.tight_layout()  # auto-fix spacing
        plt.ion()  # for live plot

    def plot(self):
        '''do live plotting'''
        sys_vars = self.session.sys_vars
        if not sys_vars['RENDER']:
            return
        ax1, p1 = self.subgraphs['total rewards']
        p1.set_ydata(
            np.append(p1.get_ydata(), sys_vars['total_r_history'][-1]))
        p1.set_xdata(np.arange(len(p1.get_ydata())))
        ax1.relim()
        ax1.autoscale_view(tight=True, scalex=True, scaley=True)

        ax1e, p1e = self.subgraphs['e']
        p1e.set_ydata(
            np.append(p1e.get_ydata(), sys_vars['explore_history'][-1]))
        p1e.set_xdata(np.arange(len(p1e.get_ydata())))
        ax1e.relim()
        ax1e.autoscale_view(tight=True, scalex=True, scaley=True)

        ax2, p2 = self.subgraphs['mean rewards']
        p2.set_ydata(np.append(p2.get_ydata(), sys_vars['mean_rewards']))
        p2.set_xdata(np.arange(len(p2.get_ydata())))
        ax2.relim()
        ax2.autoscale_view(tight=True, scalex=True, scaley=True)

        ax3, p3 = self.subgraphs['loss']
        p3.set_ydata(sys_vars['loss'])
        p3.set_xdata(np.arange(len(p3.get_ydata())))
        ax3.relim()
        ax3.autoscale_view(tight=True, scalex=True, scaley=True)

        plt.draw()
        plt.pause(0.01)

    def save(self, filename):
        '''save graph to filename'''
        self.figure.savefig(filename)


class Session(object):

    '''
    The base unit of an Experiment
    An Experiment for a config on repeat for k time
    will run k Sessions, each with identical sess_spec
    for a problem, Agent, Memory, Policy, param.
    Handles its own data, plots and saves its own graphs
    Serialized by the parent experiment_id with its session_id
    '''

    def __init__(self, experiment, session_id=0):
        self.experiment = experiment
        self.session_id = session_id
        self.sess_spec = experiment.sess_spec
        self.problem = self.sess_spec['problem']
        self.Agent = self.sess_spec['Agent']
        self.Memory = self.sess_spec['Memory']
        self.Policy = self.sess_spec['Policy']
        self.param = self.sess_spec['param']

        # init all things, so a session can only be ran once
        self.sys_vars = self.init_sys_vars()
        self.env = gym.make(self.sys_vars['GYM_ENV_NAME'])
        self.agent = self.Agent(get_env_spec(self.env), **self.param)
        self.memory = self.Memory(**self.param)
        self.policy = self.Policy(**self.param)
        self.agent.compile(self.memory, self.policy)
        logger.info('Compiled Agent, Memory, Policy')

        # data file and graph
        self.base_filename = self.experiment.base_filename + \
            '_' + str(self.session_id)
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
        if (not args.render) or mp.current_process().name != 'MainProcess':
            sys_vars['RENDER'] = False  # mute on parallel
        if environ.get('CI'):
            sys_vars['RENDER'] = False
            sys_vars['MAX_EPISODES'] = 2
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

    def debug_agent_info(self):
        logger.debug(
            "Agent info: {}".format(
                format_obj_dict(self.agent, ['learning_rate', 'n_epoch'])))
        logger.debug(
            "Memory info: size: {}".format(self.agent.memory.size()))
        logger.debug(
            "Policy info: {}".format(
                format_obj_dict(self.agent.policy, ['e'])))

    def save(self):
        '''save graph'''
        if self.sys_vars['RENDER']:
            self.grapher.save(self.graph_filename)

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
            logger.info('Problem solved? {}. At epi: {}. Params: {}'.format(
                sys_vars['solved'], sys_vars['epi'],
                pp.pformat(self.param)))
            self.env.close()

    def update_history(self):
        '''
        update the data per episode end
        '''

        sys_vars = self.sys_vars
        sys_vars['total_r_history'].append(sys_vars['total_rewards'])
        sys_vars['explore_history'].append(
            getattr(self.policy, 'e', 0) or getattr(self.policy, 'tau', 0))
        avg_len = sys_vars['REWARD_MEAN_LEN']
        # Calculating mean_reward over last 100 episodes
        # case away from np for json serializable (dumb python)
        mean_rewards = float(np.mean(sys_vars['total_r_history'][-avg_len:]))
        solved = (mean_rewards >= sys_vars['SOLVED_MEAN_REWARD'])
        sys_vars['mean_rewards'] = mean_rewards
        sys_vars['solved'] = solved

        self.grapher.plot()
        self.save()
        self.check_end()
        return sys_vars

    def run_episode(self):
        '''run ane episode, return sys_vars'''
        sys_vars = self.sys_vars
        env = self.env
        agent = self.agent

        state = env.reset()
        agent.memory.reset_state(state)
        self.debug_agent_info()
        sys_vars['total_rewards'] = 0

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
            sys_vars['total_rewards'] += reward
            if done:
                break
        self.update_history()
        return sys_vars

    def run(self):
        '''run a session of agent'''
        sys_vars = self.sys_vars
        time_start = timestamp()
        for epi in range(sys_vars['MAX_EPISODES']):
            sys_vars['epi'] = epi  # update sys_vars epi
            self.run_episode()
            if 'epi_change_learning_rate' in self.param and epi == self.param['epi_change_learning_rate']:
                self.agent.recompile_model(self.param['learning_rate'] / 10.0)
            if sys_vars['solved']:
                break

        if K._BACKEND == 'tensorflow':
            K.clear_session()  # manual gc to fix TF issue 3388
        time_end = timestamp()
        time_taken = timestamp_elapse(time_start, time_end)
        sys_vars['time_start'] = time_start
        sys_vars['time_end'] = time_end
        sys_vars['time_taken'] = time_taken

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
    - sess_spec (so we can plug in directly again to rerun)
    - summary
        - time_start
        - time_end
        - time_taken
        - metrics
    - sys_vars_array
    '''

    def __init__(self, sess_spec, times=1):
        self.sess_spec = sess_spec
        self.data = None
        self.times = times
        self.sess_spec.pop('param_range', None)  # single exp, del range
        sample_spec = stringify_param(self.sess_spec)
        self.experiment_id = '{}_{}_{}_{}_{}'.format(
            sample_spec['problem'],
            sample_spec['Agent'],
            sample_spec['Memory'],
            sample_spec['Policy'],
            timestamp()
        )
        self.base_filename = './data/{}'.format(self.experiment_id)
        self.data_filename = self.base_filename + '.json'

    def analyze(self):
        '''
        helper: analyze given data from an experiment
        return metrics
        '''
        sys_vars_array = self.data['sys_vars_array']
        mean_r_array = [sys_vars['mean_rewards']
                        for sys_vars in sys_vars_array]
        metrics = {
            'experiment_mean': np.mean(mean_r_array),
            'experiment_std': np.std(mean_r_array),
        }
        self.data['summary'].update({'metrics': metrics})
        return self.data

    def save(self):
        '''
        save the entire experiment data grid from inside run()
        '''
        with open(self.data_filename, 'w') as f:
            f.write(to_json(self.data))
        logger.info(
            'Experiment complete, written to {}'.format(self.data_filename))

    def run(self):
        '''
        helper: run a experiment for Session
        a number of times times given a sess_spec from gym_specs
        '''
        time_start = timestamp()
        sys_vars_array = []
        for i in range(self.times):
            sess = Session(experiment=self, session_id=i)
            sys_vars = sess.run()
            sys_vars_array.append(copy.copy(sys_vars))
            # update and save each time
            time_end = timestamp()
            time_taken = timestamp_elapse(time_start, time_end)

            self.data = {  # experiment data
                'sess_spec': stringify_param(self.sess_spec),
                'summary': {
                    'time_start': time_start,
                    'time_end': time_end,
                    'time_taken': time_taken,
                    'metrics': None,
                },
                'sys_vars_array': sys_vars_array,
            }
            self.analyze()
            # progressive update, write when every session is done
            self.save()
        return self.data


def run(sess_name_or_spec, times=1, param_selection=False, line_search=True):
    '''
    primary method:
    run all experiments, specified by the sess_spec or its name
    for a specified number of times per experiment
    (multiple experiments if param_selection=True)
    '''
    if isinstance(sess_name_or_spec, str):
        sess_spec = game_specs.get(sess_name_or_spec)
    else:
        sess_spec = sess_name_or_spec

    if param_selection:
        raise Exception('to be implemented, with separate py processes')
        # if line_search:
        #     param_grid = param_line_search(
        #     sess_spec['param'], sess_spec['param_range'])
        # else:
        #     param_grid = param_product(
        #         sess_spec['param'], sess_spec['param_range'])
        # sess_spec_grid = [{
        #     'problem': sess_spec['problem'],
        #     'Agent': sess_spec['Agent'],
        #     'Memory': sess_spec['Memory'],
        #     'Policy': sess_spec['Policy'],
        #     'param': param,
        # } for param in param_grid]
        # p = mp.Pool(mp.cpu_count())
        # list(p.map(
        #     partial(run_single_exp, data_grid=data_grid, times=times),
        #     sess_spec_grid))
    else:
        # run_single_exp(sess_spec, data_grid=data_grid, times=times)
        experiment = Experiment(sess_spec, times=times)
        return experiment.run()
