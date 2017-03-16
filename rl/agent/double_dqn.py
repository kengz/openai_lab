import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger, clone_model, clone_optimizer


class DoubleDQN(DQN):

    '''
    The base class of double DQNs
    '''

    def build_model(self):
        super(DoubleDQN, self).build_model()

        model_2 = clone_model(self.model)
        logger.info("Model 2 summary")
        model_2.summary()
        self.model_2 = model_2

        logger.info("Models 1 and 2 built")
        return self.model, self.model_2

    def compile_model(self):
        self.optimizer.keras_optimizer_2 = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.model.compile(
            loss='mse',
            optimizer=self.optimizer.keras_optimizer)
        self.model_2.compile(
            loss='mse',
            optimizer=self.optimizer.keras_optimizer_2)
        logger.info("Models 1 and 2 compiled")

    def compute_Q_states(self, minibatch):
        (Q_states, Q_next_states_select, _max) = super(
            DoubleDQN, self).compute_Q_states(minibatch)
        # Different from (single) dqn: Select max using model 2
        Q_next_states_max_ind = np.argmax(Q_next_states_select, axis=1)
        # same as dqn again, but use Q_next_states_max_ind above
        Q_next_states = np.clip(
            self.model_2.predict(minibatch['next_states']),
            -self.clip_val, self.clip_val)
        rows = np.arange(Q_next_states_max_ind.shape[0])
        Q_next_states_max = Q_next_states[rows, Q_next_states_max_ind]

        return (Q_states, Q_next_states, Q_next_states_max)

    def switch_models(self):
         # Switch model 1 and model 2, also the optimizers
        temp = self.model
        self.model = self.model_2
        self.model_2 = temp

        temp_optimizer = self.optimizer.keras_optimizer
        self.optimizer.keras_optimizer = self.optimizer.keras_optimizer_2
        self.optimizer.keras_optimizer_2 = temp_optimizer

    def train_an_epoch(self):
        self.switch_models()
        return super(DoubleDQN, self).train_an_epoch()
