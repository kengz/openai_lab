from rl.util import *
from rl.optimizer.base_optimizer import Optimizer
from keras.optimizers import SGD

class SGD_L(Optimizer):
    '''
    Stochastic gradient descent
    Potential params:
        lr (learning rate)
        momentum
        decay
        nesterov
    '''

    def __init__(self, param):
        super(SGD_L, self).__init__(param)
        self.set_init_optim_params()
        self.init_optimizer()
        logger.info("SGD optimizer initialized. Params: {}".format(self.optim_param))

    def set_init_optim_params(self):
        optim_param = {}
        if 'learning_rate' in self.param:
            lr = self.param['learning_rate']
            optim_param['lr'] = lr
        if 'momentum' in self.param:
            momentum = self.param['momentum']
            optim_param['momentum'] = momentum
        if 'decay' in self.param:
            decay = self.param['decay']
            optim_param['decay'] = decay
        if 'nesterov' in self.param:
            nesterov = self.param['nesterov']
            optim_param['nesterov'] = nesterov
        self.optim_param = optim_param

    def init_optimizer(self):
        self.optimizer = SGD(**self.optim_param)

    def change_optim_params(self, new_params):
        if 'learning_rate' in new_params:
            lr = new_params['learning_rate']
            self.optim_param['lr'] = lr
        if 'momemtum' in new_params:
            momentum = new_params['momentum']
            self.optim_param['momentum'] = momentum
        if 'decay' in new_params:
            decay = new_params['decay']
            self.optim_param['decay'] = decay
        if 'nesterov' in new_params:
            nesterov = new_params['nesterov']
            self.optim_param['nesterov'] = nesterov
        self.init_optimizer()
        logger.info("SGD optimizer parameters changed. New params: {}".format(self.optim_param))

         