from rl.experiment import run
from util import args

if __name__ == '__main__':
    run(args.experiment_name, **vars(args))
