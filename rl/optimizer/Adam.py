from rl.util import *
from rl.optimizer.base_optimizer import Optimizer
from keras.optimizers import Adam

class Adam_L(Optimizer):
    '''
    Adam optimizer
    Potential params:
        lr (learning rate)
        beta_1
        beta_2
        epsilon
        decay
        Suggested to leave at default params with the expected of lr
    '''

    def __init__(self, param):  
        super(Adam_L, self).__init__(param)
        self.set_init_optim_params()
        self.init_optimizer()
        logger.info("Adam optimizer initialized. Params: {}".format(self.optim_param))

    def set_init_optim_params(self):
        optim_param = {}
        if 'learning_rate' in self.param:
            lr = self.param['learning_rate']
            optim_param['lr'] = lr
        if 'beta_1' in self.param:
            beta_1 = self.param['beta_1']
            optim_param['beta_1'] = beta_1
        if 'beta_2' in self.param:
            beta_2 = self.param['beta_2']
            optim_param['beta_2'] = beta_2
        if 'epsilon' in self.param:
            epsilon = self.param['epsilon']
            optim_param['epsilon'] = epsilon
        if 'decay' in self.param:
            decay = self.param['decay']
            optim_param['decay'] = decay
        self.optim_param = optim_param

    def init_optimizer(self):
        self.optimizer = Adam(**self.optim_param)

    def change_optim_params(self, new_params):
        if 'learning_rate' in new_params:
            lr = new_params['learning_rate']
            self.optim_param['lr'] = lr
        if 'beta_1' in new_params:
            beta_1 = new_params['beta_1']
            self.optim_param['beta_1'] = momentum
        if 'beta_2' in new_params:
            beta_2 = new_params['beta_2']
            self.optim_param['beta_2'] = beta_2
        if 'epsilon' in new_params:
            epsilon = new_params['epsilon']
            self.optim_param['epsilon'] = epsilon
        if 'decay' in new_params:
            decay = new_params['decay']
            self.optim_param['decay'] = decay
        self.init_optimizer()
        logger.info("Adam optimizer parameters changed. New params: {}".format(self.optim_param))