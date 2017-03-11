import pytest
import rl
from os import environ

environ['CI'] = environ.get('CI') or 'true'


def pytest_runtest_setup(item):
    for problem in rl.util.PROBLEMS:
        if problem == 'TestPassCartPole-v0':
            pass
        else:
            rl.util.PROBLEMS[problem]['MAX_EPISODES'] = 3
