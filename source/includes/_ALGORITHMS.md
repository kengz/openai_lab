## Algorithms

General approach:
This library is intended to make it easy to run experiments with deep reinforcement learning alorithms 

Intro to RL:
Brief summary
Links

Background:
Reinforcement learning problems are characterized by incomplete information. The transition probabilities from state s to state s' given action a, for all s and a are not known. Consequently, reinforcement learning algorithms involve approximating an unknown, typically complex, non linear function. Deep neural networks are powerful function approximators and make good candidates for the function approximators in reinforcement learning algorithms.

Terminology
- Agent - encapsulates a specific algorithm. Each agent has a policy, memory, optimizer, and preprocessor
- Online training - agents are trained periodically during episodes
- Episodic training - agents are only trained after an episode has completed and before the next episode begins
- Policy - rule which determines how to act in a given state, e.g. choose the action A which has the highest Q-value in state S
- Q function - Q(S, A), estimates the value of taking action A in state S under a specific policy. 
- Q-vallue - Value of Q function for a particular S and A.
- On policy - the same policy is used to act and evaluate the quality of actions.
- Off policy - a different policy is used to act and evaluate the quality of actions. 

This library makes it easy to use and implement deep reinforcement learning algorithms. There are a number of algorithms available to use out of the box. The following sections gives an overview of all the currently implemented algorithms. Below that, we'll walk through the implementation of a single algorithm in detail: 

Q-Learning

    Q-learning algorithms attempt to estimate the optimal Q function. They are off policy algorithms in that the policy used to evaluate the target is different to the policy used to determine which state-action pairs are being visited. It is also a temporal difference algorithm. The udates  to the Q function are based on existing estimates, specifically the estimate in time t is updated using an estimate from time t+1. This allows Q-Learning algorithms to be online and incremental, so the agent can be trained during an episode. The update to Q_t(S, A) is as follows

    Q(S_t, A_t) \Leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma max_A Q(S_{t+1}, A) - Q(S_t, A)]

    For more details, please see chapter 6 of [Reinforcement Learning: An Introduction, Sutton and Barto](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

    Since the policy that is used to evaluate the target is fixed (a greedy policy that selects the action that maximises the Q-value for a particular state) and is different to the policy used to determine which state-action pairs are being visited, it is possible to use experience replay to train an agent. Agents act in environments to get experiences. Experiences consist of the state, the action taken, the next state, and the reward, and are denoted as <S_t, A_t, R_{t+1}, S_{t+1}>. These experiences are stored in the agents memory. Periodically during an episode the agent is trained. During training n batches of size m are selected from memory and the Q update step is performed. This is different to Sarsa algorithms which are on-policy and agents are trained after each experience using only the most recent experience.
    
    - Deep Q-Learning:
        - Standard Q-learning algorithm with experience replay. Online training every n experiences.

        Q update:
        Q(S, A) \Leftarrow Q(S, A) + \alpha [R + \gamma max_a Q(S', A) - Q(S, A)]

        Translation to neural network update:
        Learning rate: \alpha
        Input (x vals):  (S, A)
        Network output: Q(S, A)
        Target (y vals):  [R + \gamma max_a Q(S', A)]

        - Agents: 
            - DQN: function approximator - feedforward neural network
            - ConvDQN:  function approximator - convolutional network

    - Double Q-Learning: 
        - Q-learning algorithm with two Q function approximators to address the maximisation bias problem, Q_1, and Q_2. One Q function is used to select the action in the next state, S', the other is used to evaluate the action in state S'. Periodically the roles of each Q function are switched. Online training every n experiences.
        
        Q update(alternate between 1 and 2)
        1. Q_1(S, A) \Leftarrow Q_1(S, A) + \alpha [R + \gamma Q_2(S', argmax_A Q_1(S', A)) - Q1(S, A)]
        2. Q_2(S, A) \Leftarrow Q_2(S, A) + \alpha [R + \gamma Q_1(S', argmax_A Q_2(S', A)) - Q2(S, A)]

        Translation to neural network update:
        Learning rate: \alpha
        Input (x vals):  (S, A)
        Network output: Q_1(S, A) or Q_2(S, A)  
        Target (y vals):  [R + \gamma Q_1(S', argmax_A Q_2(S', A))] or [R + \gamma Q_2(S', argmax_A Q_1(S', A))]

        - Agents
            - DoubleDQN: function approximator - feedforward neural network
            - DoubleConvQN:  function approximator - convolutional network

    - Deep Q-Learning with weight freezing: 
        - Deep Q-Learning algorithms tends to be unstable. To address this issue, create two Q function approximators, one for exploration, Q_e, and one for evaluating the target, Q_t. The target is a copy of the exploration network with frozen weights which lag the exploration network. These weights are updated periodically to match the exploration network. Freezing the target network weights help avoids oscillations in the policy, where slight changes to Q-values can lead to significant changes in the policy, and helps break correlations between the Q-network and the target. See [David Silver's](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf) lecture slides for more details. Online training every n experiences.

        Q update:
        Q_e(S, A) \Leftarrow Q_e(S, A) + \alpha [R + \gamma max_A Q_t(S', A) - Q_e(S, A)]
        Periodically set Q_t = Q_e (e.g. after every episode)

        Translation to neural network update:
        Learning rate: \alpha
        Input (x vals):  (S, A)
        Network output: Q_e(S, A)
        Target (y vals):  [R + \gamma max_A Q_t(S', A)]
        Update is to Q_e

         - Agents
            - DQNFreeze: function approximator - feedforward neural network
            - ConvQNFreeze:  function approximator - convolutional network

Sarsa

    Sarsa algorithms also attempt to estimate the optimal Q function. They are on policy algorithms in that the policy used to evaluate the target is the same as to the policy used to determine which state-action pairs are being visited. Like Q-Learning, Sarsa is also a temporal difference algorithm. The sarsa update is as follows.

    Select A_{t+1} in state S_{t+1} using policy
    Q(S_t, A_t) \Leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A)]

    This update is made each time the agent acts in an environment and gets an experience <S_t, A_t, R_{t+1}, S_{t+1}>

    - Deep Sarsa




   - Off Policy Expected Sarsa (experimental): 
        - Sarsa is typically an on policy algorithm. However, if a different policy is used to evaluate the target than the one used to explore, it becomes and off-policy algorithm.  With this set up, Q-Learning can be understood as a specific instance of Off Policy Expected Sarsa, when the policy used to evaluate the target is the greedy policy.

Policy Gradient : Coming soon


Agent: detailed view



This library makes it 

- RL with function approx. See chapter 12 of Sutton and Barto book. Each RL algo has at least on func approx component. The function approx is a neural network 
- Use Keras but not limited to

An OpenAI Lab algorithm consists of:
1. Overarching Reinforcement Learning algorithm, Q-Learning, 
2. Function approximator: neural network architecture and hyperparameters
3. State preprcessing (optional)
4. Memory

Implementing your own algorithm:

- Subclass base agent
- Neural networks need an input and a target. Guide to structure, one action, etc. 




