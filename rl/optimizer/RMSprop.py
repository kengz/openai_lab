from rl.util import *
from rl.optimizer.base_optimizer import Optimizer
from keras.optimizers import RMSprop

class RMSprop_L(Optimizer):
    '''
    RMS prop
    Potential params:
        lr (learning rate)
        rho
        decay
        epsilon
    '''

    def __init__(self, param):
        super(RMSprop_L, self).__init__(param)
        self.set_init_optim_params()
        self.init_optimizer()
        logger.info("RMS Prop optimizer initialized. Params: {}".format(self.optim_param))

    def set_init_optim_params(self):
        optim_param = {}
        if 'learning_rate' in self.param:
            lr = self.param['learning_rate']
            optim_param['lr'] = lr
        if 'rho' in self.param:
            rho = self.param['rho']
            optim_param['rho'] = rho
        if 'decay' in self.param:
            decay = self.param['decay']
            optim_param['decay'] = decay
        if 'epsilon' in self.param:
            epsilon = self.param['epsilon']
            optim_param['epsilon'] = epsilon
        self.optim_param = optim_param

    def init_optimizer(self):
        self.optimizer = RMSprop(**self.optim_param)

    def change_optim_params(self, new_params):
        if 'learning_rate' in new_params:
            lr = new_params['learning_rate']
            self.optim_param['lr'] = lr
        if 'rho' in new_params:
            rho = new_params['rho']
            self.optim_param['rho'] = rho
        if 'decay' in new_params:
            decay = new_params['decay']
            self.optim_param['decay'] = decay
        if 'epsilon' in new_params:
            epsilon = new_params['epsilon']
            self.optim_param['epsilon'] = epsilon
        self.init_optimizer()
        logger.info("RMS Prop optimizer parameters changed. New params: {}".format(self.optim_param))