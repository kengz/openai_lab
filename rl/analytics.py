import matplotlib
import numpy as np
import pandas as pd
import platform
import seaborn as sns
import warnings
from os import environ
from rl.util import *

warnings.filterwarnings("ignore", module="matplotlib")

# TODO fix mp breaking on Mac shit,
# except when running -b with agg backend
# (no GUI rendered,but saves graphs)
# set only if it's not MacOS
if environ.get('CI') or platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'agg'
else:
    matplotlib.rcParams['backend'] = 'TkAgg'
sns.set(style="whitegrid", color_codes=True,
        rc={'lines.linewidth': 1.0, 'backend': matplotlib.rcParams['backend']})

STATS_COLS = [
    'mean_rewards_per_epi_stats_mean',
    'mean_rewards_stats_mean',
    'epi_stats_mean',
    'solved_ratio_of_sessions',
    'max_total_rewards_stats_mean',
    't_stats_mean',
    'experiment_id'
]
EXPERIMENT_GRID_Y_COLS = ['mean_rewards_per_epi_stats_mean']


class Grapher(object):

    '''
    Grapher object that belongs to a Session
    to draw graphs from its data
    '''

    def __init__(self, session):
        if not args.plot_graph:
            return
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'  # mute matplotlib toolbar
        self.plt = plt
        self.session = session
        self.graph_filename = self.session.graph_filename
        self.subgraphs = {}
        self.figure = self.plt.figure(facecolor='white', figsize=(8, 9))
        self.figure.suptitle(wrap_text(self.session.session_id))
        self.init_figure()

    def init_figure(self):
        if environ.get('CI'):
            return
        # graph 1
        ax1 = self.figure.add_subplot(
            311,
            frame_on=False,
            title="\n\ntotal rewards per episode",
            ylabel='total rewards')
        p1, = ax1.plot([], [])
        self.subgraphs['total rewards'] = (ax1, p1)

        ax1e = ax1.twinx()
        ax1e.set_ylabel('exploration rate').set_color('r')
        ax1e.set_frame_on(False)
        ax1e.grid(False)
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

        self.plt.tight_layout()  # auto-fix spacing
        self.plt.ion()  # for live plot

    def plot(self):
        '''do live plotting'''
        if not args.plot_graph or environ.get('CI'):
            return
        sys_vars = self.session.sys_vars
        ax1, p1 = self.subgraphs['total rewards']
        p1.set_ydata(
            sys_vars['total_rewards_history'])
        p1.set_xdata(np.arange(len(p1.get_ydata())))
        ax1.relim()
        ax1.autoscale_view(tight=True, scalex=True, scaley=True)

        ax1e, p1e = self.subgraphs['e']
        p1e.set_ydata(
            sys_vars['explore_history'])
        p1e.set_xdata(np.arange(len(p1e.get_ydata())))
        ax1e.relim()
        ax1e.autoscale_view(tight=True, scalex=True, scaley=True)

        ax2, p2 = self.subgraphs['mean rewards']
        p2.set_ydata(
            sys_vars['mean_rewards_history'])
        p2.set_xdata(np.arange(len(p2.get_ydata())))
        ax2.relim()
        ax2.autoscale_view(tight=True, scalex=True, scaley=True)

        ax3, p3 = self.subgraphs['loss']
        p3.set_ydata(sys_vars['loss'])
        p3.set_xdata(np.arange(len(p3.get_ydata())))
        ax3.relim()
        ax3.autoscale_view(tight=True, scalex=True, scaley=True)

        self.plt.draw()
        self.plt.pause(0.01)
        self.save()

    def save(self):
        '''save graph to filename'''
        self.figure.savefig(self.graph_filename)


def basic_stats(array):
    '''generate the basic stats for a numerical array'''
    if not len(array):
        return None
    return {
        'min': np.min(array).astype(float),
        'max': np.max(array).astype(float),
        'mean': np.mean(array).astype(float),
        'std': np.std(array).astype(float),
    }


def compose_data(experiment):
    '''
    compose raw data from an experiment object
    into useful summary and full metrics for analysis
    '''
    sys_vars_array = experiment.data['sys_vars_array']

    # collect all data from sys_vars_array
    solved_sys_vars_array = list(filter(
        lambda sv: sv['solved'], sys_vars_array))
    mean_rewards_array = np.array(list(map(
        lambda sv: sv['mean_rewards'], sys_vars_array)))
    max_total_rewards_array = np.array(list(map(
        lambda sv: np.max(sv['total_rewards_history']), sys_vars_array)))
    epi_array = np.array(list(map(lambda sv: sv['epi'], sys_vars_array)))
    mean_rewards_per_epi_array = np.divide(mean_rewards_array, epi_array)
    t_array = np.array(list(map(lambda sv: sv['t'], sys_vars_array)))
    time_taken_array = np.array(list(map(
        lambda sv: timestamp_elapse_to_seconds(sv['time_taken']),
        sys_vars_array)))
    solved_epi_array = np.array(list(map(
        lambda sv: sv['epi'], solved_sys_vars_array)))
    solved_t_array = np.array(list(map(
        lambda sv: sv['t'], solved_sys_vars_array)))
    solved_time_taken_array = np.array(list(map(
        lambda sv: timestamp_elapse_to_seconds(sv['time_taken']),
        solved_sys_vars_array)))

    # compose sys_vars stats
    stats = {
        'num_of_sessions': len(sys_vars_array),
        'solved_num_of_sessions': len(solved_sys_vars_array),
        'solved_ratio_of_sessions': float(len(
            solved_sys_vars_array)) / experiment.times,
        'mean_rewards_stats': basic_stats(mean_rewards_array),
        'mean_rewards_per_epi_stats': basic_stats(
            mean_rewards_per_epi_array),
        'max_total_rewards_stats': basic_stats(max_total_rewards_array),
        'epi_stats': basic_stats(epi_array),
        't_stats': basic_stats(t_array),
        'time_taken_stats': basic_stats(time_taken_array),
        'solved_epi_stats': basic_stats(solved_epi_array),
        'solved_t_stats': basic_stats(solved_t_array),
        'solved_time_taken_stats': basic_stats(solved_time_taken_array),
    }

    # summary metrics
    metrics = {
        'mean_rewards_per_epi_stats_mean': stats[
            'mean_rewards_per_epi_stats']['mean'],
        'mean_rewards_stats_mean': stats['mean_rewards_stats']['mean'],
        'epi_stats_mean': stats['epi_stats']['mean'],
        'solved_ratio_of_sessions': stats['solved_ratio_of_sessions'],
        'max_total_rewards_stats_mean': stats[
            'max_total_rewards_stats']['mean'],
        't_stats_mean': stats['t_stats']['mean'],
    }

    # param variables for independent vars of experiments
    param_variables = {
        pv: experiment.sess_spec['param'][pv] for
        pv in experiment.param_variables}
    param_variables = flat_cast_dict(param_variables)

    experiment.data['metrics'].update(metrics)
    experiment.data['param_variables'] = param_variables
    experiment.data['stats'] = stats
    return experiment.data


# plot the experiment_grid data from data_df
# X are columns with name starting with 'variable_'
# Y cols are defined below
def plot_experiment_grid(data_df, experiment_id):
    if len(data_df) < 2:  # no multi selection
        return
    prefix_id = prefix_id_from_experiment_id(experiment_id)
    X_cols = list(filter(lambda c: c.startswith('variable_'), data_df.columns))
    for x in X_cols:
        for y in EXPERIMENT_GRID_Y_COLS:
            df_plot = sns.swarmplot(x=x, y=y, data=data_df)
            fig = df_plot.get_figure()
            fig.suptitle(wrap_text(prefix_id))
            filename = './data/{}/experiment_grid_plot_{}_vs_{}.png'.format(
                prefix_id, x, y)
            fig.savefig(filename)


def analyze_data(experiment_grid_data_or_prefix_id):
    '''
    get all the data from all experiments.run()
    or read from all data files matching the prefix of experiment_id
    e.g. usage without running:
    prefix_id = 'DevCartPole-v0_DQN_LinearMemoryWithForgetting_BoltzmannPolicy_2017-01-15_142810'
    analyze_data(prefix_id)
    '''
    if isinstance(experiment_grid_data_or_prefix_id, str):
        experiment_grid_data = load_data_array_from_prefix_id(
            experiment_grid_data_or_prefix_id)
    else:
        experiment_grid_data = experiment_grid_data_or_prefix_id

    stats_array, param_variables_array = [], []
    for data in experiment_grid_data:
        stats = flatten_dict(data['stats'])
        stats.update({'experiment_id': data['experiment_id']})
        stats_array.append(stats)
        param_variables = data['param_variables']
        param_variables_array.append(param_variables)

    raw_stats_df = pd.DataFrame.from_dict(stats_array)
    stats_df = raw_stats_df[STATS_COLS]

    param_variables_df = pd.DataFrame.from_dict(param_variables_array)
    param_variables_df.columns = [
        'variable_'+c for c in param_variables_df.columns]

    data_df = pd.concat([stats_df, param_variables_df], axis=1)
    for c in data_df.columns:
        if data_df[c].dtype == object:  # guard
            data_df[c] = data_df[c].astype('category')

    data_df.sort_values(
        ['mean_rewards_per_epi_stats_mean'],
        inplace=True, ascending=False)

    experiment_id = experiment_grid_data[0]['experiment_id']
    save_experiment_grid_data(data_df, experiment_id)
    plot_experiment_grid(data_df, experiment_id)
    return data_df
