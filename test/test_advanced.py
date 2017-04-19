import unittest
import pytest
from os import environ
from rl.experiment import run
from . import conftest
import pandas as pd


class AdvancedTest(unittest.TestCase):

    @classmethod
    def test_sarsa(cls):
        data_df = run('rand_sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_exp_sarsa(cls):
        data_df = run('exp_sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_offpol_sarsa(cls):
        data_df = run('offpol_sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_acrobot(cls):
        data_df = run('acrobot')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_mountain_dqn(cls):
        data_df = run('mountain_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_lunar_dqn(cls):
        data_df = run('lunar_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @unittest.skipIf(environ.get('CI'),
                     "Delay CI test until dev stable")
    @classmethod
    def test_breakout_dqn(cls):
        data_df = run('breakout_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @unittest.skipIf(environ.get('CI'),
                     "Delay CI test until dev stable")
    @classmethod
    def test_breakout_double_dqn(cls):
        data_df = run('breakout_double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_cartpole_ac_argmax(cls):
        data_df = run('cartpole_ac_argmax')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_pendulum_ddpg(cls):
        data_df = run('pendulum_ddpg')
        assert isinstance(data_df, pd.DataFrame)
