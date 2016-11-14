from rl.agent.double_dqn import DoubleDQN
from rl.policy import TargetedEpsilonGreedyPolicy
from rl.util import logger, pp
from keras.models import Sequential
from keras.layers.core import Dense


class LunarDoubleDQN(DoubleDQN):

    def __init__(self, *args, **kwargs):
        super(LunarDoubleDQN, self).__init__(*args, **kwargs)
        # change the policy
        self.policy = TargetedEpsilonGreedyPolicy(self)

    def build_net(self):
        logger.info(pp.pformat(self.env_spec))
        model = Sequential()
        model.add(Dense(8,
                        input_shape=(self.env_spec['state_dim'],),
                        init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(6, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(6, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        logger.info("Model 1 summary")
        model.summary()
        self.model = model

        model2 = Sequential.from_config(model.get_config())
        logger.info("Model 2 summary")
        model2.summary()
        self.model2 = model2
        return model, model2
