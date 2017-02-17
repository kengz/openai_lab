from rl.optimizer.base_optimizer import Optimizer
from keras.optimizers import SGD


class SGDOptimizer(Optimizer):

    '''
    Stochastic gradient descent
    Potential param:
        lr (learning rate)
        momentum
        decay
        nesterov
    '''

    def __init__(self, **kwargs):
        self.optim_param_keys = ['lr', 'momentum', 'decay', 'nesterov']
        super(SGDOptimizer, self).__init__(**kwargs)

    def init_optimizer(self):
        self.keras_optimizer = SGD(**self.optim_param)
