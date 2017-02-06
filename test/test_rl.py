import unittest
import pandas as pd
from os import environ
environ['CI'] = environ.get('CI') or 'true'
from rl.experiment import run


class DQNTest(unittest.TestCase):

    def test_gym_tour(self):
        data_df = run('dummy')
        assert isinstance(data_df, pd.DataFrame)

    def test_q_table(self):
        data_df = run('q_table')
        assert isinstance(data_df, pd.DataFrame)

    def test_dqn(self):
        data_df = run('dqn')
        assert isinstance(data_df, pd.DataFrame)

    def test_double_dqn(self):
        data_df = run('double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    def test_dev_dqn_pass(self):
        data_df = run('dev_dqn')
        max_total_rewards = data_df['max_total_rewards_stats_mean'][0]
        print(max_total_rewards)
        assert max_total_rewards > 100
