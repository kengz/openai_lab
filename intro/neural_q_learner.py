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
# - build e-greedy action and e-update rule
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

    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.init_e = 1.0
        self.final_e = 0.1
        self.e = self.init_e
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

    def update_e(self, epi, t):
        '''
        strategy to update epsilon
        sawtooth wave pattern that decreases in an apisode, and across episodes
        epi = replay_memory.epi
        t = replay_memory.t
        local_e = the e in this episode
        global_e = the global scaling e
        '''
        global_half_life = 5.
        local_half_life = 10.
        # local_e = self.init_e * math.exp(-.693/float(half_life*t))
        global_e = self.init_e * math.exp(-.693/global_half_life*float(epi))
        local_e = self.init_e * math.exp(-.693/local_half_life*float(t))
        compound_e = local_e * global_e
        # rescaled, translated
        print(local_e, global_e, compound_e)
        self.e = compound_e*abs(self.init_e - self.final_e) + self.final_e
        return self.e

    def e_greedy_action():
        # need a decay policy for epi
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
        epi = replay_memory.get_ep()
        # replay_memory used to guide annealing too
        # also it shd be step wise, epxosed
        self.m.fit(X, Y)  # self-batching, set Y on the fly, etc
        # self.m.save('models/dqn.tfl')

q = DQN(get_env_spec(env))
print(q.env_specs['state_dim'])
epi = 0
t = 0
e = q.update_e(epi, t)
print(e)


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
