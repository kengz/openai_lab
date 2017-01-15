import numpy as np
from rl.util import logger
from rl.agent.dqn import DQN
from rl.util import logger
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
K.set_image_dim_ordering('tf')

class ConvDQN(DQN):

    def __init__(self, *args, **kwargs):
        super(ConvDQN, self).__init__(*args, **kwargs)

    def build_hidden_layers(self, model):
        '''
        build the hidden layers into model using parameter self.hidden_layers
        '''
        model.add(Convolution2D(self.hidden_layers[0][0],
                                self.hidden_layers[0][1],
                                self.hidden_layers[0][2],
                                subsample=self.hidden_layers[0][3],
                                input_shape=(self.env_spec['state_dim']),
                                activation=self.hidden_layers_activation,
                                init='lecun_uniform'))

        if (len(self.hidden_layers) > 1):
            for i in range(1, len(self.hidden_layers)):
                model.add(Convolution2D(self.hidden_layers[i][0],
                                self.hidden_layers[i][1],
                                self.hidden_layers[i][2],
                                subsample=self.hidden_layers[i][3],
                                activation=self.hidden_layers_activation,
                                init='lecun_uniform'))
        model.add(Flatten())
        model.add(Dense(256, init='lecun_uniform', activation=self.hidden_layers_activation))
        return model

    def build_model(self):
        '''
        Optimizer is RMSProp, default learning rate used
        '''

        model = Sequential()
        self.build_hidden_layers(model)
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))

        logger.info("Model summary")
        model.summary()
        self.model = model

        self.optimizer = RMSprop()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Model built and compiled")
        return self.model

    def recompile_model(self, new_lr):
        '''
        Option to change model optimizer settings
        Doesn't do anything at the moment
        '''
        self.learning_rate = new_lr
        self.optimizer = RMSprop()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Model recompiled but optimizer set to RMSProp so it didn't do anything")