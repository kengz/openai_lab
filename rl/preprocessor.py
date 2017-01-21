import numpy as np
import scipy as sp
from rl.util import logger, log_self


# Util functions for state preprocessing

def resize_image(im):
    return sp.misc.imresize(im, (110, 84))


def crop_image(im):
    return im[-84:, :]


def process_image_atari(im):
    # Image preprocessing from the paper
    # Playing Atari with Deep Reinforcement Learning, 2013
    # Takes an RGB image and converts it to grayscale,
    # downsizes to 110 x 84
    # and crops to square 84 x 84, taking bottomost rows of image
    im_gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
    im_resized = resize_image(im_gray)
    im_cropped = crop_image(im_resized)
    return im_cropped


def create_dummy_states(state):
    state_shape = state.state_shape
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

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        '''
        Construct externally, and set at Agent.compile()
        '''
        self.agent = None
        self.state = None
        self.exp_queue = []
        self.state_buffer = ()

    def exp_queue_size(self):
        return len(self.exp_queue)

    def reset_state(self, init_state):
        '''
        reset the state of LinearMemory per episode env.reset()
        '''
        self.state = init_state
        (previous_state, pre_previous_state,
            pre_pre_previous_state) = create_dummy_states(self.state)
        self.previous_state = previous_state
        self.pre_previous_state = pre_previous_state
        self.pre_pre_previous_state = pre_pre_previous_state

    def preprocess_state(self):
        raise NotImplementedError()

    def advance_state(self, next_state):
        # TODO can be generalized into a memory queue
        self.pre_pre_previous_state = self.pre_previous_state
        self.pre_previous_state = self.previous_state
        self.previous_state = self.state
        self.state = next_state

    def add_exp(self, action, reward, next_state, done):
        # Buffer currently set to hold only last 4 experiences
        # Amount needed for Atari games preprocessing
        self.exp_queue.append([self.state, action, reward, next_state, done])
        if (len(exp_queue) > 4):
            del exp_queue[0]
        self.advance_state(next_state)

    def preprocess_memory(self, action, reward, next_state, done):
        raise NotImplementedError()


class NoPreProcessor(PreProcessor):

    '''
    Default class, no preprocessing
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking):
        super(NoPreProcessor, self).__init__()
        log_self(self)

    def preprocess_state(self):
        return self.state

    def preprocess_memory(self, action, reward, next_state, done):
        # No state processing
        self.add_exp(action, reward, next_state, done)
        (state, action, reward, next_state, done) = self.exp_queue[-1]
        # TODO this is a bit odd, since now we have add_exp_processed and
        # add_exp for memory
        self.agent.memory.add_exp_processed(
            state, action, reward,
            next_state, next_state, done)


class StackStates(PreProcessor):

    '''
    Current and last state are concatenated to form input to model
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking):
        super(StackStates, self).__init__()
        log_self(self)

    def preprocess_state(self):
        return np.concatenate([self.previous_state, self.state])

    def preprocess_memory(self, action, reward, next_state, done):
        # Concatenates previous + current states
        self.add_exp(action, reward, next_state, done)
        if (self.exp_queue_size() < 1):  # insufficient queue
            return
        (state, action, reward, next_state, done) = self.exp_queue[-1]
        processed_state = np.concatenate([self.exp_queue[-2][0], state])
        processed_next_state = np.concatenate([state, next_state])
        if (self.exp_queue_size() == 1):
            logger.debug("State shape: {}".format(processed_state.shape))
            logger.debug(
                "Next state shape: {}".format(processed_next_state.shape))
        self.agent.memory.add_exp_processed(
            processed_state, action, reward,
            # TODO ensure we want to add processed_next_state as state
            processed_next_state, processed_next_state, done)


class DiffStates(PreProcessor):

    '''
    Different between current and last state is input to model
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking):
        super(DiffStates, self).__init__()
        log_self(self)

    def preprocess_state(self):
        return self.state - self.previous_state

    def preprocess_memory(self, action, reward, next_state, done):
        # Change in state params, curr_state - last_state
        self.add_exp(action, reward, next_state, done)
        if (self.exp_queue_size() < 1):  # insufficient queue
            return
        (state, action, reward, next_state, done) = self.exp_queue[-1]
        processed_state = state - exp_queue[-2][0]
        processed_next_state = next_state - state
        if (self.exp_queue_size() == 1):
            logger.debug("State shape: {}".format(processed_state.shape))
            logger.debug(
                "Next state shape: {}".format(processed_next_state.shape))
        self.agent.memory.add_exp_processed(
            processed_state, action, reward,
            # TODO ensure we want to add processed_next_state as state
            processed_next_state, processed_next_state, done)


class Atari(PreProcessor):

    '''
    Convert images to greyscale, downsize, crop, then stack 4 states
    NOTE: Image order is cols * rows * channels to match openai gym format
    Input to model is rows * cols * channels (== states)
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking):
        super(Atari, self).__init__()
        log_self(self)

    def preprocess_state(self):
        state_arrays = (process_image_atari(self.state),
                        process_image_atari(self.previous_state),
                        process_image_atari(self.pre_previous_state),
                        process_image_atari(self.pre_pre_previous_state))
        return np.stack(state_arrays, axis=-1)

    def preprocess_memory(self, action, reward, next_state, done):
        self.add_exp(action, reward, next_state, done)
        if (self.exp_queue_size() < 3):  # insufficient queue
            return
        (state, action, reward, next_state, done) = self.exp_queue[-1]
        state_arrays = (process_image_atari(exp_queue[-1][0]),
                        process_image_atari(exp_queue[-2][0]),
                        process_image_atari(exp_queue[-3][0]),
                        process_image_atari(exp_queue[-4][0]))
        next_state_arrays = (process_image_atari(exp_queue[-1][3]),
                             process_image_atari(exp_queue[-2][3]),
                             process_image_atari(exp_queue[-3][3]),
                             process_image_atari(exp_queue[-4][3]))
        processed_state = np.stack(state_arrays, axis=-1)
        processed_next_state = np.stack(next_state_arrays, axis=-1)
        done = exp_queue[-1][4]
        if (self.exp_queue_size() == 3):
            logger.debug("State shape: {}".format(processed_state.shape))
            logger.debug(
                "Next state shape: {}".format(processed_next_state.shape))
        self.agent.memory.add_exp_processed(
            processed_state, action, reward,
            # TODO ensure we want to add processed_next_state as state
            processed_next_state, processed_next_state, done)
