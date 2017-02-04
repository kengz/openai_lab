from rl.experiment import run
from util import args

if __name__ == '__main__':
    run(args.sess_name, **vars(args))
