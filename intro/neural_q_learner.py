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

    def __init__(self, env_spec):
        self.env_spec = env_spec
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

    def add_exp(self, action, reward, next_state, terminal):
        '''
        after the env.step(a) that returns s', r,
        using the previously stored state for the s,
        form an experience tuple <s, a, r, s'>
        '''
        exp = dict(zip(
            ['state', 'action', 'reward', 'next_state', 'terminal'],
            [deepcopy(self.state), action, reward, next_state, terminal]))
        # store exp, update state and time
        self.memory.append(exp)
        self.state = next_state
        self.t += 1
        return exp

    def get_exp(self, index):
        return deepcopy(self.memory[index])

    def one_hot_action(self, action):
        action_arr = np.zeros(self.env_spec['action_dim'])
        action_arr[action] = 1
        return action_arr

    def format_minibatch(self, exp_batch):
        '''
        transpose, transform the minibatch into useful form
        '''
        minibatch = dict(zip(
            ['states', 'actions', 'rewards', 'next_states', 'terminals'],
            [
                np.array([exp['state'] for exp in exp_batch]),
                np.array([self.one_hot_action(exp['action'])
                          for exp in exp_batch]),
                np.array([exp['reward'] for exp in exp_batch]),
                np.array([exp['next_state'] for exp in exp_batch]),
                np.array([exp['terminal'] for exp in exp_batch])
            ]
        ))
        return minibatch

    def rand_minibatch(self):
        '''
        get a minibatch of random exp for training
        '''
        memory_size = len(self.memory)
        if memory_size <= BATCH_SIZE:
            # to prevent repetition and initial overfitting
            rand_inds = np.random.permutation(memory_size)
        else:
            rand_inds = np.random.randint(
                memory_size, size=BATCH_SIZE)
        exp_batch = [self.get_exp(i) for i in rand_inds]
        minibatch = self.format_minibatch(exp_batch)
        return minibatch


class DQN(object):

    def __init__(self, env_spec, session):
        self.env_spec = env_spec
        self.session = session
        self.INIT_E = 1.0
        self.FINAL_E = 0.05
        self.e = self.INIT_E
        self.EPI_HALF_LIFE = 100.
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.build_graph()

    def build_net(self):
        # X = tflearn.input_data(shape=[None, self.env_spec['state_dim']])
        # reshape into 3D tensor for conv
        # net = tf.reshape(X, [-1, self.env_spec['state_dim'], 1])
        # net = tflearn.conv_1d(net, 8, 2, activation='relu')
        # net = tflearn.fully_connected(net, 16, activation='relu')
        # net = tflearn.dropout(net, 0.5)
        # no conv
        X = tflearn.input_data(shape=[None, self.env_spec['state_dim']])
        net = tflearn.fully_connected(X, 8, activation='relu')
        # net = tflearn.dropout(net, 0.5)
        # net = tflearn.fully_connected(net, 8, activation='relu')
        # net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(
            net, self.env_spec['action_dim'])
        # aight output is the q_values
        self.X = X
        self.net = net
        return net

    def build_graph(self):
        net = self.build_net()
        self.Y = tf.placeholder("float", [None, self.env_spec['action_dim']])
        self.loss = tf.reduce_mean(tf.square(self.net - self.Y))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        return self.train_op

    def train(self, replay_memory):
        '''
        step 1,2,3,4 of algo
        '''
        self.update_e(replay_memory)
        minibatch = replay_memory.rand_minibatch()
        # algo step 1
        Q_states = self.net.eval(feed_dict={self.X: minibatch['states']})
        # algo step 2
        Q_next_states = self.net.eval(
            feed_dict={self.X: minibatch['next_states']})
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        # Q targets for batch-actions a;
        # with terminal to make future reward 0 if end
        Q_targets_a = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_next_states_max
        # set Q_targets of a as above, and the non-action units' Q_targets to
        # as from algo step 1
        Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
            (1 - minibatch['actions']) * Q_states

        _, loss = self.session.run([self.train_op, self.loss], feed_dict={
            self.X: minibatch['states'],
            self.Y: Q_targets,
        })
        return loss

    def update_e(self, replay_memory):
        '''
        strategy to update epsilon
        '''
        unscaled_e = self.INIT_E * \
            math.exp(-.693/self.EPI_HALF_LIFE*float(len(replay_memory.memory)))
        # rescale to fit in 0.1 to 1.0, translated + 0.1
        self.e = unscaled_e*abs(self.INIT_E - self.FINAL_E) + self.FINAL_E
        return self.e

    def select_action(self, state):
        '''
        step 1 of algo, feedforward
        '''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
        else:
            Q_state = self.net.eval(
                feed_dict={self.X: [state]}, session=self.session)
            action = self.session.run(tf.argmax(Q_state, 1))[0]
        return action


# update the hisory, max len = MAX_HISTORY
# @return [bool] solved
def update_history(total_rewards, epi, total_t):
    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)
    logs = [
        '{:->20}'.format(''),
        'Episode {}'.format(epi),
        'Finished at t={}'.format(total_t),
        'Average reward for the last {} episodes: {}'.format(
            MAX_HISTORY, mean_rewards),
        'Reward for this episode: {}'. format(total_rewards)
    ]
    print('\n'.join(logs))
    solved = mean_rewards >= SOLVED_MEAN_REWARD
    return solved


# run an episode
# @return [bool] if the problem is solved by this episode
def run_episode(epi, env, replay_memory, dqn):
    total_rewards = 0
    state = env.reset()
    replay_memory.reset_state(state)
    for t in range(MAX_STEPS):
        env.render()
        action = dqn.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_memory.add_exp(action, reward, next_state, int(done))
        dqn.train(replay_memory)
        state = next_state
        total_rewards += reward
        if done:
            break
    solved = update_history(total_rewards, epi, t)
    return solved


# the primary method to run
# epi starts from 1 to MAX_EPISODES (inclusive)
# @return [bool] if the problem is solved
def deep_q_learn(env):
    sess = tf.InteractiveSession()
    env_spec = get_env_spec(env)
    replay_memory = ReplayMemory(env_spec)
    dqn = DQN(env_spec, sess)
    init = tf.initialize_all_variables()
    sess.run(init)
    for epi in range(MAX_EPISODES):
        solved = run_episode(epi, env, replay_memory, dqn)
        if solved:
            break
    print('Problem solved? {}'.format(solved))
    return solved


if __name__ == '__main__':
    deep_q_learn(env)
