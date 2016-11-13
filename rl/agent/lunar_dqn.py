from rl.agent.dqn import DQN
from rl.policy import OscillatingEpsilonGreedyPolicy
from rl.util import logger, pp
from keras.models import Sequential
from keras.layers.core import Dense


class LunarDQN(DQN):

    def __init__(self, *args, **kwargs):
        super(LunarDQN, self).__init__(*args, **kwargs)
        # change the policy
        self.policy = OscillatingEpsilonGreedyPolicy(self)

    def build_net(self):
        logger.info(pp.pformat(self.env_spec))
        model = Sequential()
        # Not clear how much better the algorithm is with regularization
        model.add(Dense(8,
                        input_shape=(self.env_spec['state_dim'],),
                        init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(6, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        model.summary()
        self.model = model
        return model
