from keras.optimizers import SGD

class SGD(Optimizer):
    '''
    Stochastic gradient descent
    Potential params:
        lr (learning rate)
        momentum
        decay
        nesterov
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking
        super(SGD, self).__init__()

    def init_optimizer():
        pass

    def change_optim_params(self, new_params):
        pass
         