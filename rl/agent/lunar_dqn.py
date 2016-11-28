from rl.agent.dqn import DQN
from rl.policy import TargetedEpsilonGreedyPolicy
from rl.util import logger
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


class LunarDQN(DQN):

    def __init__(self, *args, **kwargs):
        super(LunarDQN, self).__init__(*args, **kwargs)
        # change the policy
        self.policy = TargetedEpsilonGreedyPolicy(self)

    def build_model(self):
        model = Sequential()
        if (len(self.hidden_layers) == 1):
            model.add(Dense(self.hidden_layers[0],
                            input_shape=(self.env_spec['state_dim'],),
                            init='lecun_uniform', activation=self.hidden_layers_activation))
        else:
            model.add(Dense(self.hidden_layers[0],
                            input_shape=(self.env_spec['state_dim'],),
                            init='lecun_uniform', activation=self.hidden_layers_activation))
            for i in range(1, len(self.hidden_layers)):
                model.add(Dense(self.hidden_layers[i], 
                    init='lecun_uniform', activation=self.hidden_layers_activation))
                
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        model.summary()
        self.model = model

        self.optimizer = SGD(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Model built and compiled")
        return self.model
