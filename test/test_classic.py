import unittest
import pytest
from os import environ
from rl.experiment import run
from . import conftest
import pandas as pd


class ClassicTest(unittest.TestCase):

    @classmethod
    def test_quickstart_dqn(cls):
        data_df = run('quickstart_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dqn_epsilon(cls):
        data_df = run('dqn_epsilon')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dqn(cls):
        data_df = run('dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_double_dqn(cls):
        data_df = run('double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_sarsa(cls):
        data_df = run('sarsa')
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
    def test_cartpole_ac_argmax(cls):
        data_df = run('cartpole_ac_argmax')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dqn_v1(cls):
        data_df = run('dqn_v1')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_acrobot(cls):
        data_df = run('acrobot')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_pendulum_ddpg_linearnoise(cls):
        data_df = run('pendulum_ddpg_linearnoise')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_mountain_dqn(cls):
        data_df = run('mountain_dqn')
        assert isinstance(data_df, pd.DataFrame)
