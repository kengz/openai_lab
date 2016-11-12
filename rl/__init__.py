# curse of python pathing, hack to solve rel import
import sys
from os import path
file_path = path.normpath(path.join(path.dirname(__file__)))
sys.path.insert(0, file_path)

# another py curse, expose to prevent 'rl.rl.<method>' call
from rl import main, replay_memory, util
