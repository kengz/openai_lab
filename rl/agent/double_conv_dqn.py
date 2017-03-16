from rl.agent.conv_dqn import ConvDQN
from rl.agent.double_dqn import DoubleDQN


class DoubleConvDQN(DoubleDQN, ConvDQN):

    '''
    The base class of double convolutional DQNs
    extended from DoubleDQN and ConvDQN
    multiple inheritance will use the method from the first class
    if multiple ones exists
    '''

    def build_hidden_layers(self, model):
        ConvDQN.build_hidden_layers(self, model)
