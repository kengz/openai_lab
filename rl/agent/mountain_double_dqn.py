from agent.double_dqn import DoubleDQN
from util import logger, pp
from keras.models import Sequential
from keras.layers.core import Dense


class MountainDoubleDQN(DoubleDQN):

    def build_net(self):
        logger.info(pp.pformat(self.env_spec))
        model = Sequential()
        model.add(Dense(2,
                        input_shape=(self.env_spec['state_dim'],),
                        init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(3, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(4, init='lecun_uniform', activation='sigmoid'))
        model.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        logger.info("Model 1 summary")
        model.summary()
        self.model = model

        model2 = Sequential()
        model2.add(Dense(2,
                         input_shape=(self.env_spec['state_dim'],),
                         init='lecun_uniform', activation='sigmoid'))
        model2.add(Dense(3, init='lecun_uniform', activation='sigmoid'))
        model2.add(Dense(4, init='lecun_uniform', activation='sigmoid'))
        model2.add(Dense(self.env_spec['action_dim'], init='lecun_uniform'))
        logger.info("Model 2 summary")
        model2.summary()
        self.model2 = model2
        return model, model2

    def update_e(self, sys_vars, replay_memory):
        '''strategy to update epsilon'''
        super(MountainDoubleDQN, self).update_e(sys_vars, replay_memory)
        epi = sys_vars['epi']
        if not (epi % 3) and epi > 10:
            # drop to 1/3 of the current exploration rate
            self.e = max(self.e/3., self.final_e)
        return self.e
