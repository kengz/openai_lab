import unittest
import pandas as pd
from os import environ
environ['CI'] = environ.get('CI') or 'true'
from rl.experiment import run
from rl.util import *
import rl.util
import os
import json

PATH = os.path.dirname(__file__)
rl.util.PROBLEMS = json.loads(open(
os.path.join(PATH, 'test_problems.json')).read())
rl.util.EXPERIMENT_SPECS = json.loads(open(
os.path.join(PATH, 'test_experiment_specs.json')).read())
for experiment_name in rl.util.EXPERIMENT_SPECS:
    rl.util.EXPERIMENT_SPECS[experiment_name]['experiment_name'] = experiment_name
    if 'param_range' not in rl.util.EXPERIMENT_SPECS[experiment_name]:
        continue
    param_range = rl.util.EXPERIMENT_SPECS[experiment_name]['param_range']
    for param_key, param_val in param_range.items():
        param_range[param_key] = sorted(param_val)
    rl.util.EXPERIMENT_SPECS[experiment_name]['param_range'] = param_range

class DQNTest(unittest.TestCase):

    @classmethod
    def test_gym_tour(cls):
        data_df = run('test_dummy')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_q_table(cls):
        data_df = run('test_q_table')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dqn(cls):
        data_df = run('test_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_double_dqn(cls):
        data_df = run('test_double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dqn_freeze(cls):
        data_df = run('test_dqn_freeze')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_sarsa(cls):
        data_df = run('test_sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_sarsa_exp(cls):
        data_df = run('test_sarsa_exp')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_sarsa_offpol(cls):
        data_df = run('test_sarsa_offpol')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_mountain_dqn(cls):
        data_df = run('test_mountain_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_lunar_dqn(cls):
        data_df = run('test_lunar_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_conv_dqn(cls):
        data_df = run('test_conv_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_double_conv_dqn(cls):
        data_df = run('test_double_conv_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dev_dqn_pass(cls):
        data_df = run('test_test_dqn')
        max_total_rewards = data_df['max_total_rewards_stats_mean'][0]
        print(max_total_rewards)
        assert max_total_rewards > 50

