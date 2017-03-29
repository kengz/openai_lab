# <a name="algorithms"></a>Algorithms

The currently implemented algorithms combine deep neural networks with a number of classic reinforcement learning algorithms. These are just a starting point. Please invent your own! We will also continue to add algorithms over time. 

## What is reinforcement learning?

*Reinforcement learning (RL) is learning from interaction with an environment, from the consequences of action, rather than from explicit teaching. RL become popular in the 1990s within machine learning and artificial intelligence, but also within operations research and with offshoots in psychology and neuroscience.*

*Most RL research is conducted within the mathematical framework of Markov decision processes (MDPs). MDPs involve a decision-making agent interacting with its environment so as to maximize the cumulative reward it receives over time. The agent perceives aspects of the environment's state and selects actions. The agent may estimate a value function and use it to construct better and better decision-making policies over time.*

*RL algorithms are methods for solving this kind of problem, that is, problems involving sequences of decisions in which each decision affects what opportunities are available later, in which the effects need not be deterministic, and in which there are long-term goals. RL methods are intended to address the kind of learning and decision making problems that people and animals face in their normal, everyday lives*
        *- Rich Sutton*

For further reading on reinforcement learning see [David Silver's lectures](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) and the book, Reinforcement Learning: An Introduction, by Sutton and Barto.

## Why are deep neural networks useful for solving RL problems?
RL problems are characterized by incomplete information. The transition probabilities from one state to another given the action taken, for all states and actions are not known. So in order to solve problems, RL algorithms involve approximating one or more unknown, typically complex, non linear functions. Deep neural networks make good candidates for these function approximators since they excel at approximating complex functions, particularly if the states are characterized by pixel level features.

## Terminology
- Agent: encapsulates a specific algorithm. Each agent has a policy, memory, optimizer, and preprocessor
- Online training: agents are trained periodically during episodes
- Episodic training: agents are only trained after an episode has completed and before the next episode begins
- Policy: rule which determines how to act in a given state, e.g. choose the action A which has the highest Q-value in state S. May be deterministic or stochatic. 
- Q function: Q(S, A), estimates the value of taking action A in state S under a specific policy. 
- Q-vallue: Value of Q function for a particular S and A.
- On policy: the same policy is used to act and evaluate the quality of actions.
- Off policy: a different policy is used to act and evaluate the quality of actions. 

## Currently implemented algorithms

### Q-Learning

Q-learning algorithms attempt to estimate the optimal Q function, i.e thevalue of taking action A in state S under a specific policy. Q-learning algorithms have an implicit policy, typically \epsilon-greedy in which the action with the maximum Q value is selected with probability(1 - \epsilon) and a random action is taken with probability \epsilon. The random actions encourage exploration of the state space and help prevent algorithms from getting stuck in local minima. Q-learning algorithms are off-policy algorithms in that the policy used to evaluate the value of the action taken is different to the policy used to determine which state-action pairs are visited. It is also a temporal difference algorithm. Updates  to the Q function are based on existing estimates. The estimate in time t is updated using an estimate from time t+1. This allows Q-Learning algorithms to be online and incremental, so the agent can be trained during an episode. The update to Q_t(S, A) is as follows

$$ Q(S_t, A_t) \Leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma max_A Q(S_{t+1}, A) - Q(S_t, A)] $$

For more details, please see chapter 6 of [Reinforcement Learning: An Introduction, Sutton and Barto](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

Since the policy that is used to evaluate the target is fixed (a greedy policy that selects the action that maximises the Q-value for a particular state) and is different to the policy used to determine which state-action pairs are being visited, it is possible to use experience replay to train an agent. This is often needed to Agents act in environments to get experiences. Experiences consist of the state, the action taken, the next state, and the reward, and are denoted as <S_t, A_t, R_{t+1}, S_{t+1}>. These experiences are stored in the agents memory. Periodically during an episode the agent is trained. During training n batches of size m are selected from memory and the Q update step is performed. This is different to Sarsa algorithms which are on-policy and agents are trained after each experience using only the most recent experience.
    
#### Deep Q-Learning:
Standard Q-learning algorithm with experience replay. Online training every n experiences.

Q update:
$$ Q(S, A) \Leftarrow Q(S, A) + \alpha [R + \gamma max_a Q(S', A) - Q(S, A)] $$

Translation to neural network update:
Learning rate: \alpha
Input (x vals):  (S, A)
Network output: Q(S, A)
Target (y vals):  [R + \gamma max_a Q(S', A)]

Agents: 
- DQN: function approximator - feedforward neural network
- ConvDQN:  function approximator - convolutional network

#### Double Q-Learning: 
Q-learning algorithm with two Q function approximators to address the maximisation bias problem, Q_1, and Q_2. One Q function is used to select the action in the next state, S', the other is used to evaluate the action in state S'. Periodically the roles of each Q function are switched. Online training every n experiences.

Q update(alternate between 1 and 2)
1. $ Q_1(S, A) \Leftarrow Q_1(S, A) + \alpha [R + \gamma Q_2(S', argmax_A Q_1(S', A)) - Q1(S, A)] $
2. $Q_2(S, A) \Leftarrow Q_2(S, A) + \alpha [R + \gamma Q_1(S', argmax_A Q_2(S', A)) - Q2(S, A)] $

Translation to neural network update:
Learning rate: \alpha
Input (x vals):  (S, A)
Network output: Q_1(S, A) or Q_2(S, A)  
Target (y vals):  [R + \gamma Q_1(S', argmax_A Q_2(S', A))] or [R + \gamma Q_2(S', argmax_A Q_1(S', A))]

Agents
- DoubleDQN: function approximator - feedforward neural network
- DoubleConvQN:  function approximator - convolutional network

#### Deep Q-Learning with weight freezing: 
Deep Q-Learning algorithms tends to be unstable. To address this issue, create two Q function approximators, one for exploration, Q_e, and one for evaluating the target, Q_t. The target is a copy of the exploration network with frozen weights which lag the exploration network. These weights are updated periodically to match the exploration network. Freezing the target network weights help avoids oscillations in the policy, where slight changes to Q-values can lead to significant changes in the policy, and helps break correlations between the Q-network and the target. See [David Silver's](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf) lecture slides for more details. Online training every n experiences.

Q update:
$$Q_e(S, A) \Leftarrow Q_e(S, A) + \alpha [R + \gamma max_A Q_t(S', A) - Q_e(S, A)]$$
Periodically set $Q_t = Q_e$ (e.g. after every episode) or $Q_t = (1 - \epsilon)Q_t + \epsilon Q_e$

Translation to neural network update:
Learning rate: \alpha
Input (x vals):  $(S, A)$
Network output: $Q_e(S, A)$
Target $(y vals):  [R + \gamma max_A Q_t(S', A)]$
Update is to $$

 Agents
- FreezeDQN: function approximator - feedforward neural network

### Sarsa

Sarsa algorithms also attempt to estimate the optimal Q function. They are on policy algorithms so the policy used to evaluate the target is the same as to the policy used to determine which state-action pairs are being visited. Like Q-Learning, Sarsa is a temporal difference algorithm. However, since they are on policy, it is trickier to take advantage of experience replay, requiring storage of the action in state t+1 and the Q-value for the state and action selected in t+1 in an experience. In the following implementations, updates are made after each action with the exception of off policy expected Sarsa.

Sarsa update:
Select $A_{t+1}$ in state $S_{t+1}$ using policy
$Q(S_t, A_t) \Leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A)]$

This update is made each time the agent acts in an environment and gets an experience <S_t, A_t, R_{t+1}, S_{t+1}>

#### Deep Sarsa
Standard Sarsa algortthm

Q update (same as Q-Learning):
$$ Q(S, A) \Leftarrow Q(S, A) + \alpha [R + \gamma max_a Q(S', A) - Q(S, A)] $$

Translation to neural network update:
Learning rate: \alpha
Input (x vals):  (S, A)
Network output: Q(S, A)
Target (y vals):  [R + \gamma max_a Q(S', A)]

Agents
- DeepSarsa: function approximator - feedforward neural network

#### Deep Expected Sarsa
Uses the expected value of the Q function under the current policy to construct the target instead of the Q-value for the action selected.

Q update (same and Q-Learning):
$$ Q(S, A) \Leftarrow Q(S, A) + \alpha [R + \gamma E_a Q(S', a) - Q(S, A)] $$

Translation to neural network update:
Learning rate: \alpha
Input (x vals):  (S, A)
Network output: Q(S, A)
Target (y vals):  [R + \gamma E_a Q(S', a)]

Agents
- DeepExpectedSarsa: function approximator - feedforward neural network


#### Off Policy Expected Sarsa (experimental): 
Sarsa is typically an on policy algorithm. However, if a different policy is used to evaluate the target than the one used to explore, it becomes and off-policy algorithm.  With this set up, Q-Learning can be understood as a specific instance of Off Policy Expected Sarsa, when the policy used to evaluate the target is the greedy policy.

Q update and translation to neural network update: Same as DQN with fixed epsilon.

Agents
- OffPolicySarsa: function approximator - feedforward neural network

### Policy Gradient 

#### Deep Deterministic Policy Gradients:  In progress







