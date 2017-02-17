from rl.optimizer.base_optimizer import Optimizer
from keras.optimizers import RMSprop


class RMSpropOptimizer(Optimizer):

    '''
    RMS prop
    Potential param:
        lr (learning rate)
        rho
        decay
        epsilon
    '''

    def __init__(self, **kwargs):
        self.optim_param_keys = ['lr', 'rho', 'decay', 'epsilon']
        super(RMSpropOptimizer, self).__init__(**kwargs)

    def init_optimizer(self):
        self.keras_optimizer = RMSprop(**self.optim_param)
