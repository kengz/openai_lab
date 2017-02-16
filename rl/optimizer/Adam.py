from keras.optimizers import Adam

class Adam(Optimizer):
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

    def __init__(self, **kwargs):  # absorb generic param without breaking
        super(Adam, self).__init__()

    def init_optimizer():
        pass

    def change_optim_params(self, new_params):
        pass