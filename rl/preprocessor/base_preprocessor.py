import numpy as np
from rl.util import logger, log_self


def create_dummy_states(state):
    state_shape = state.shape
    previous_state = np.zeros(state_shape)
    pre_previous_state = np.zeros(state_shape)
    pre_pre_previous_state = np.zeros(state_shape)
    if (previous_state.ndim == 1):
        previous_state = np.zeros([state_shape[0]])
        pre_previous_state = np.zeros([state_shape[0]])
        pre_pre_previous_state = np.zeros([state_shape[0]])
    return (previous_state, pre_previous_state, pre_pre_previous_state)


class PreProcessor(object):

    '''
    The Base class for state preprocessing
    '''

    def __init__(self, max_queue_size=4, **kwargs):
        '''Construct externally, and set at Agent.compile()'''
        self.env_spec = None  # set from preprocess_env_spec
        self.agent = None
        self.state = None
        self.exp_queue = []
        self.MAX_QUEUE_SIZE = max_queue_size
        self.never_debugged = True
        log_self(self)

    def reset_state(self, init_state):
        '''reset the state of LinearMemory per episode env.reset()'''
        self.state = np.array(init_state)  # cast into np for safety
        (previous_state, pre_previous_state,
            pre_pre_previous_state) = create_dummy_states(self.state)
        self.previous_state = previous_state
        self.pre_previous_state = pre_previous_state
        self.pre_pre_previous_state = pre_pre_previous_state
        return self.preprocess_state()

    def exp_queue_size(self):
        return len(self.exp_queue)

    def debug_state(self, processed_state, processed_next_state):
        if self.never_debugged:
            logger.debug("State shape: {}".format(processed_state.shape))
            logger.debug(
                "Next state shape: {}".format(processed_next_state.shape))
            self.never_debugged = False

    def preprocess_env_spec(self, env_spec):
        '''helper to tweak env_spec according to preprocessor'''
        class_name = self.__class__.__name__
        if class_name is 'StackStates':
            env_spec['state_dim'] = env_spec['state_dim'] * 2
        elif class_name is 'Atari':
            env_spec['state_dim'] = (84, 84, 4)
        self.env_spec = env_spec
        return env_spec

    def preprocess_state(self):
        raise NotImplementedError()

    def preprocess_action(self, action):
        '''
        generalize continuous to act on discrete space too
        current implementation is by picking the strongest action
        action = array [a_1, a_2, ...]

        Usually in the discrete case these are picked by
        the highest Q(s, a_m) -> a_m, then get index m/one-hot encoding
        Continuous agent will output [a_1, a_2, ...] by not as an one-hot encoding,
        but an array of real values, allowing for simultaneous and real-valued actions

        To reduce this to discrete,
        take the analogy of pressing all buttons on the gaming console (discrete) at once,
        and only the strongest action gets registered.
        So, pick np.argmax([a_1, a_2, ...]), but save the real valued arrays in memory for training.
        '''
        if self.env_spec['actions'] == 'continuous':
            # continuous problem, keep as is
            assert action.shape == (self.env_spec['action_dim'], )
            processed_action = action
        else:  # discrete problem
            if np.shape(action) == (self.env_spec['action_dim'], ):
                # action is from continuous agent (array), pick max
                # forces positive value to be strength of action
                processed_action = np.argmax(action)
            else:  # already suited for discrete
                processed_action = action
        return processed_action

    def advance_state(self, next_state):
        self.pre_pre_previous_state = self.pre_previous_state
        self.pre_previous_state = self.previous_state
        self.previous_state = self.state
        self.state = next_state

    def add_raw_exp(self, action, reward, next_state, done):
        '''
        Buffer currently set to hold only last 4 experiences
        Amount needed for Atari games preprocessing
        '''
        self.exp_queue.append([self.state, action, reward, next_state, done])
        if (self.exp_queue_size() > self.MAX_QUEUE_SIZE):
            del self.exp_queue[0]
        self.advance_state(next_state)

    def preprocess_memory(self, action, reward, next_state, done):
        raise NotImplementedError()
