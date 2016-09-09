import tflearn
import tensorflow as tf

# next:
# - build e-greedy action and e-update rule
# - net with placeholders, loss
# - DQN.select_action
# - DQN.train


class DQN(object):

    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.e = 1.0
        self.learning_rate = 0.001
        # this can be inferred from replay memory, or not. replay memory shall
        # be over all episodes,

    # !need to get episode, and game step of current episode

    def build(self, loss):
        # see extending tensorflow cnn for placeholder usage
        net = tflearn.input_data(shape=[None, self.env_specs['state_dim']])
        net = tflearn.conv_1d(net, 32, 2, activation='relu')
        net = tflearn.conv_1d(net, 32, 2, activation='relu')
        net = tflearn.conv_1d(net, 64, 2, activation='relu')
        net = tflearn.conv_1d(net, 64, 2, activation='relu')
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(
            net, self.env_specs['action_dim'], activation='softmax')
        net = tflearn.regression(net, optimizer='rmsprop',
                                 loss=loss, learning_rate=self.learning_rate)
        m = tflearn.DNN(self.net, tensorboard_verbose=3)
        # prolly need to use tf.placeholder for Y of loss
        self.m = m
        return self.m

    def update_e(self):
        '''
        strategy to update epsilon
        '''
        self.e
        return

    def e_greedy_action():
        # need a decay policy for ep
        return

    def select_action(next_state):
        '''
        step 1 of algo
        '''
        return

    def train(replay_memory):
        '''
        step 2,3,4 of algo
        '''
        rand_mini_batch = replay_memory.rand_exp_batch()
        ep = replay_memory.get_ep()
        # replay_memory used to guide annealing too
        # also it shd be step wise, epxosed
        self.m.fit(X, Y)  # self-batching, set Y on the fly, etc
        # self.m.save('models/dqn.tfl')

# print(DQN(env).env_specs['state_dim'])
