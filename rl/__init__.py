# curse of python pathing, hack to solve rel import
import sys
from os import path
file_path = path.normpath(path.join(path.dirname(__file__)))
sys.path.insert(0, file_path)

# another py curse, expose to prevent 'agent.<agent>' call
from os.path import dirname, basename, isfile
import glob
pattern = "/*.py"
modules = glob.glob(dirname(__file__)+pattern)
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]
