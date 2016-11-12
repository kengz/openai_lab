from agent.dqn import DQN
from util import logger, pp
from keras.models import Sequential
from keras.layers.core import Dense


class LunarDQN(DQN):

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

    def update_e(self, sys_vars, replay_memory):
        '''strategy to update epsilon'''
        super(LunarDQN, self).update_e(sys_vars, replay_memory)
        epi = sys_vars['epi']
        if not (epi % 2) and epi > 15:
            # drop to 1/3 of the current exploration rate
            self.e = max(self.e/3., self.final_e)
        return self.e
