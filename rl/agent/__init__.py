# curse of python pathing, hack to solve rel import
import sys
from os import path
file_path = path.normpath(path.join(path.dirname(__file__)))
sys.path.insert(0, file_path)

# another py curse, expose to prevent 'agent.<agent>' call
from agent import double_dqn, dqn, dummy, lunar_double_dqn, lunar_dqn, q_table
