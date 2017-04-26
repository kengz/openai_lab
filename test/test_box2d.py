import unittest
import pytest
from os import environ
from rl.experiment import run
from . import conftest
import pandas as pd


class Box2DTest(unittest.TestCase):

    @classmethod
    def test_lunar_dqn(cls):
        data_df = run('lunar_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_lunar_double_dqn(cls):
        data_df = run('lunar_double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_walker_ddpg_linearnoise(cls):
        data_df = run('walker_ddpg_linearnoise')
        assert isinstance(data_df, pd.DataFrame)
