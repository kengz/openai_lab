'''
discounted future reward, with discount factor γ
R[t] = r[t] + γ r[t+1] + γ^2 r[t+2] + ...
R[t] = r[t] + γ R[t+1]
define function to calc best score as end game after performing action a in state s,
i.e. max discount future reward
Q(s_t, a_t) = max_π R[t+1]

where π is the policy, to yield the highest Q val of all actions
s.t. π(s) = argmax_a Q(s, a)

obtain by Bellman equation
Q(s, a) = r + γ max_a' Q(s', a')

can approx iteratively using table
by carry out a, obs reward r and new state s', with max a'
Q(s, a) = Q(s, a) - α Q(s, a) + α(r + γ max_a' Q(s', a'))

The above can be done by a simple table,
but we can use a neural net to represent Q function,
taking s, a, outputting single Q(s,a), or even better, 
taking s (observation space dim), outputting Q(s, a) for a in A (number of output units same a num_actions)
then output (real values) can be regressed, optimized with simple square loss:
L = 0.5*(r + γ max_a' Q(s', a') - Q(s,a))^2
inside the () is target minus prediction

Algo:
given a transition <s, a, r, s'>, do update Q by:
1. Do a feedforward pass with s to get predicted Q vals for all actions in A
2. Do a feedforward pass for next state s' and calc max over all output units max_a' Q(s', a')
3. Set Q-value TARGET (like target Y) for action a to r + γ max_a' Q(s', a'). For other actions, set TARGET as the same returned from (1.), i.e. with 0 errors.
4. update weights using backprop


Experience replay:
all experience <s, a, r, s'> need to be stored in replay memory
Then training, pick random samples instead of the latest
to break the similarity of subsequent training that would drive into local minimum

Exploration-exploitation:
do ε-greedy exploration - with prob ε to choose random action, else go with greedy action with highest Q-value
decrease ε over time from 1 to 0.1

Deep Q-learning algo:
init replay memory D
init action-value fn Q with random weights
observe initial state s
repeat
  select an action a
    with prob ε select a random action
    otherwise select a = argmax_a' Q(s, a') (step 1. above)
  carry out a
  observe reward r and new state s'
  store exp <s, a, r, s'> in replay memory D

  sample random transitions <ss, aa, rr, ss'> from D
  calculate target for each minibatch transition
    if ss' is terminal state then tt = rr
    otherwise set tt = rr + γ max_a' Q(s', a') (from step 2, 3 above)
  train the Q network using (tt - Q(ss, aa))^2 as loss (backprop step 4 above)

  s = s'

until terminated

p/s there are more tricks, target network, error clipping, reward clipping etc.
'''


import gym
import tflearn
import tensorflow as tf
import numpy as np
from copy import deepcopy

env = gym.make('CartPole-v0')
# need state_dim, output_dim, also range, bounds


def get_env_dims(env):
    '''
    helper to get the env dimensions
    '''
    return {
        'state_dim': env.observation_space.shape[0],
        'state_bounds': np.transpose(
            [env.observation_space.low, env.observation_space.high]),
        'action_dim': env.action_space.n
    }


class DQN(object):

    def __init__(self, env):
        self.env_dims = get_env_dims(env)
        self.learning_rate = 0.001

    def build(self, loss):
        # see extending tensorflow cnn for placeholder usage
        net = tflearn.input_data(shape=[None, self.env_dims['state_dim']])
        net = tflearn.conv_1d(net, 32, 2, activation='relu')
        net = tflearn.conv_1d(net, 32, 2, activation='relu')
        net = tflearn.conv_1d(net, 64, 2, activation='relu')
        net = tflearn.conv_1d(net, 64, 2, activation='relu')
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(
            net, self.env_dims['action_dim'], activation='softmax')
        net = tflearn.regression(net, optimizer='rmsprop',
                                 loss=loss, learning_rate=self.learning_rate)
        m = tflearn.DNN(self.net, tensorboard_verbose=3)
        # prolly need to use tf.placeholder for Y of loss
        self.m = m
        return self.m

    def e_greedy_action():
        return

    def select_action(next_state):
        '''
        step 1 of algo
        '''
        return

    def train(exp, rand_exp_batch):
        '''
        step 2,3,4 of algo
        '''
        # also it shd be step wise, epxosed
        self.m.fit(X, Y)  # self-batching, set Y on the fly, etc
        # self.m.save('models/dqn.tfl')

# print(DQN(env).env_dims['state_dim'])


class ReplayMemory(object):

    def __init__(self, init_state, batch_size):
        self.state = init_state
        self.memory = []
        self.batch_size = batch_size
        self.memory_size = 0

    def add_exp(self, action, reward, next_state):
        '''
        after the env.step(a) that returns s', r,
        using the previously stored state for the s,
        form an experience tuple <s, a, r, s'>
        '''
        exp = dict(zip(['state', 'action', 'reward', 'next_state'],
                       [deepcopy(self.state), action, reward, next_state]))
        # store and move the pointer
        self.memory.append(exp)
        self.memory_size += 1
        self.state = next_state
        return exp

    def get_exp(index):
        return deepcopy(self.memory[index])

    def rand_exp_batch(self):
        '''
        get a minibatch of randon exp for training
        '''
        if self.memory_size <= self.batch_size:
            # to prevent repetition and initial overfitting
            rand_inds = np.random.permutation(self.memory_size)
        else:
            rand_inds = np.random.randint(
                self.memory_size, size=self.batch_size)
        exp_batch = [self.get_exp(i) for i in rand_inds]
        return exp_batch


def deep_q_learn(env):
    # q = DQN(env)
    next_state = env.reset()
    total_rewards = 0
    replay_memory = ReplayMemory.new(next_state)
    for t in range(MAX_STEPS):
        env.render()
        # action = q.select_action(next_state)
        next_state, reward, done, info = env.step(action)
        exp = replay_memory.add_exp(action, reward, next_state)
        rand_exp_batch = replay_memory.rand_exp_batch()
        # q.train(exp, rand_exp_batch)  # calc target, shits, train backprop
        total_rewards += reward
        if done:
            break
    return
