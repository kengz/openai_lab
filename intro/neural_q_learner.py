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
import math
import tflearn
import tensorflow as tf
import numpy as np
from copy import deepcopy
from collections import deque

MAX_STEPS = 200
SOLVED_MEAN_REWARD = 195.0
MAX_EPISODES = 1000
MAX_HISTORY = 100
episode_history = deque(maxlen=MAX_HISTORY)
BATCH_SIZE = 32

env = gym.make('CartPole-v0')


# next:
# / e-update rule
# / build e-greedy action and
# - net with placeholders, loss
# - DQN.select_action
# - DQN.train


def get_env_spec(env):
    '''
    return the env specs: dims, actions
    '''
    return {
        'state_dim': env.observation_space.shape[0],
        'state_bounds': np.transpose(
            [env.observation_space.low, env.observation_space.high]),
        'action_dim': env.action_space.n,
        'actions': list(range(env.action_space.n))
    }


class ReplayMemory(object):

    def __init__(self):
        self.memory = []
        self.state = None
        # set to -1 so the first post-increment calc can use val = 0
        self.epi = -1
        self.t = -1

    def reset_state(self, init_state):
        '''
        reset the state of ReplayMemory per episode env.reset()
        '''
        self.state = init_state
        self.epi += 1
        self.t = -1

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
        self.state = next_state
        self.t += 1
        return exp

    def get_exp(self, index):
        return deepcopy(self.memory[index])

    def get_ep(self):
        '''
        return the number of episode recorded so far
        '''
        return self.epi

    def rand_exp_batch(self):
        '''
        get a minibatch of randon exp for training
        '''
        memory_size = len(self.memory)
        if memory_size <= BATCH_SIZE:
            # to prevent repetition and initial overfitting
            rand_inds = np.random.permutation(memory_size)
        else:
            rand_inds = np.random.randint(
                memory_size, size=BATCH_SIZE)
        exp_batch = [self.get_exp(i) for i in rand_inds]
        return exp_batch


class DQN(object):

    def __init__(self, env_spec):
        self.env_spec = env_spec
        self.INIT_E = 1.0
        self.FINAL_E = 0.1
        self.e = self.INIT_E
        self.EPI_HALF_LIFE = 20.
        self.T_HALF_LIFE = float(MAX_STEPS)/6.
        self.learning_rate = 0.001
        # this can be inferred from replay memory, or not. replay memory shall
        # be over all episodes,

    # !need to get episode, and game step of current episode

    def build_net(self):
        X = tf.placeholder(tf.float32, shape=[None, self.env_spec['state_dim']], name='X')
        X = tf.reshape(X, [-1, self.env_spec['state_dim'], 1])
        net = tflearn.conv_1d(X, 32, 2, activation='relu')
        net = tflearn.conv_1d(net, 32, 2, activation='relu')
        net = tflearn.conv_1d(net, 64, 2, activation='relu')
        net = tflearn.conv_1d(net, 64, 2, activation='relu')
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(
            net, self.env_spec['action_dim'], activation='softmax')
        # aight output is the q_values
        return net

    def build_graph(self):
        net = self.build_net()
        # boolean at index as action number
        a = tf.placeholder(
            tf.float32, [None, self.env_spec['action_dim']], name='a')
        # a is boolean, reduce to single Q value
        action_q_values = tf.reduce_sum(tf.mul(net, a), reduction_indices=1)
        Y = tf.placeholder(tf.float32, [None], name='y')
        loss = tflearn.mean_square(action_q_values, Y)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)),
            tf.float32), name='acc')
        # min_op = optimizer.minimize(loss, var_list=network_params)

        trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=BATCH_SIZE)
        trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=3)
        self.trainer = trainer
        return self.trainer


    def update_e(self, epi, t):
        '''
        strategy to update epsilon
        sawtooth wave pattern that decreases in an apisode, and across episodes
        epi = replay_memory.epi
        t = replay_memory.t
        local_e = the e in this episode
        global_e = the global scaling e
        '''
        global_e = self.INIT_E * math.exp(-.693/self.EPI_HALF_LIFE*float(epi))
        local_e = self.INIT_E * math.exp(-.693/self.T_HALF_LIFE*float(t))
        compound_e = local_e * global_e
        # rescaled, translated
        print(local_e, global_e, compound_e)
        self.e = compound_e*abs(self.INIT_E - self.FINAL_E) + self.FINAL_E
        return self.e

    def best_action(self, state):
        return False

    def select_action(self, next_state):
        '''
        step 1 of algo
        '''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
        else:
            action = self.best_action(state)
        return action

    def train(self, replay_memory):
        '''
        step 2,3,4 of algo
        '''
        rand_mini_batch = replay_memory.rand_exp_batch()
        epi = replay_memory.get_ep()
        # replay_memory used to guide annealing too
        # also it shd be step wise, epxosed
        self.trainer.fit({X: trainX, Y: trainY})  # self-batching, set Y on the fly, etc
        # self.m.save('models/dqn.tfl')

q = DQN(get_env_spec(env))
# print(q.env_spec['state_dim'])
# epi = 30
# t = 60
# e = q.update_e(epi, t)
e = q.select_action(2)
print(e)
# g = q.build_net()
g = q.build_graph()
print(g)


# update the hisory, max len = MAX_HISTORY
# @return [bool] solved
def update_history(total_rewards, epi, total_t):
    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)
    logs = [
        'Episode {}'.format(epi),
        'Finished at t={}'.format(total_t),
        'Average reward for the last {} episodes: {}'.format(
            MAX_HISTORY, mean_rewards),
        'Reward for this episode: {}'. format(total_rewards)
    ]
    print(logs.join('\n'))
    solved = mean_rewards >= SOLVED_MEAN_REWARD
    return solved


# run an episode
# @return [bool] if the problem is solved by this episode
def run_episode(epi, env, replay_memory, q):
    total_rewards = 0
    next_state = env.reset()
    replay_memory.reset_state(next_state)
    for t in range(MAX_STEPS):
        env.render()
        action = q.select_action(next_state)
        next_state, reward, done, info = env.step(action)
        exp = replay_memory.add_exp(action, reward, next_state)
        # q.train(replay_memory)  # calc target, shits, train backprop
        total_rewards += reward
        if done:
            break
    solved = update_history(total_rewards, epi, t)
    return solved


# the primary method to run
# epi starts from 1 to MAX_EPISODES (inclusive)
# @return [bool] if the problem is solved
def deep_q_learn(env):
    env_spec = get_env_spec(env)
    replay_memory = ReplayMemory()
    q = DQN(env_spec)
    for epi in range(MAX_EPISODES):
        solved = run_episode(epi, env, replay_memory, q)
        if solved:
            break
    print('Problem solved? {}'.format(solved))
    return solved
