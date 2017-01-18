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

    def preprocess_action_sel(self, state,
                              previous_state,
                              pre_previous_state,
                              pre_pre_previous_state):
        raise NotImplementedError()

    def preprocess_memory(self, temp_exp_mem, t):
        raise NotImplementedError()


class NoPreProcessor(PreProcessor):

    '''
    Default class, no preprocessing
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking):
        super(NoPreProcessor, self).__init__()
        log_self(self)

    def preprocess_action_sel(self, state,
                              previous_state,
                              pre_previous_state,
                              pre_pre_previous_state):
        return state

    def preprocess_memory(self, temp_exp_mem, t):
        # No state processing
        state = temp_exp_mem[-1][0]
        action = temp_exp_mem[-1][1]
        reward = temp_exp_mem[-1][2]
        next_state = temp_exp_mem[-1][3]
        done = temp_exp_mem[-1][4]
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

    def preprocess_action_sel(self, state,
                              previous_state,
                              pre_previous_state,
                              pre_pre_previous_state):
        return np.concatenate([previous_state, state])

    def preprocess_memory(self, temp_exp_mem, t):
        # Concatenates previous + current states
        if (t >= 1):
            processed_state = np.concatenate(
                [temp_exp_mem[-2][0], temp_exp_mem[-1][0]])
            action = temp_exp_mem[-1][1]
            reward = temp_exp_mem[-1][2]
            processed_next_state = np.concatenate(
                [temp_exp_mem[-1][0], temp_exp_mem[-1][3]])
            next_state = processed_next_state
            done = temp_exp_mem[-1][4]
            if (t == 1):
                logger.debug("State shape: {}".format(processed_state.shape))
                logger.debug(
                    "Next state shape: {}".format(processed_next_state.shape))
            self.agent.memory.add_exp_processed(
                processed_state, action, reward,
                processed_next_state, next_state, done)


class DiffStates(PreProcessor):

    '''
    Different between current and last state is input to model
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking):
        super(DiffStates, self).__init__()
        log_self(self)

    def preprocess_action_sel(self, state,
                              previous_state,
                              pre_previous_state,
                              pre_pre_previous_state):
        return state - previous_state

    def preprocess_memory(self, temp_exp_mem, t):
        # Change in state params, curr_state - last_state
        if (t >= 1):
            processed_state = temp_exp_mem[-1][0] - temp_exp_mem[-2][0]
            action = temp_exp_mem[-1][1]
            reward = temp_exp_mem[-1][2]
            processed_next_state = temp_exp_mem[-1][3] - temp_exp_mem[-1][0]
            next_state = processed_next_state
            done = temp_exp_mem[-1][4]
            if (t == 1):
                logger.debug("State shape: {}".format(processed_state.shape))
                logger.debug(
                    "Next state shape: {}".format(processed_next_state.shape))
            self.agent.memory.add_exp_processed(
                processed_state, action, reward,
                processed_next_state, next_state, done)


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

    def preprocess_action_sel(self, state,
                              previous_state,
                              pre_previous_state,
                              pre_pre_previous_state):
        arrays = (process_image_atari(state),
                  process_image_atari(previous_state),
                  process_image_atari(pre_previous_state),
                  process_image_atari(pre_pre_previous_state))
        return np.stack(arrays, axis=-1)

    def preprocess_memory(self, temp_exp_mem, t):
        if (t >= 3):
            arrays = (process_image_atari(temp_exp_mem[-1][0]),
                      process_image_atari(temp_exp_mem[-2][0]),
                      process_image_atari(temp_exp_mem[-3][0]),
                      process_image_atari(temp_exp_mem[-4][0]))
            next_arrays = (process_image_atari(temp_exp_mem[-1][3]),
                           process_image_atari(temp_exp_mem[-2][3]),
                           process_image_atari(temp_exp_mem[-3][3]),
                           process_image_atari(temp_exp_mem[-4][3]))
            processed_state = np.stack(arrays, axis=-1)
            action = temp_exp_mem[-1][1]
            reward = temp_exp_mem[-1][2]
            processed_next_state = np.stack(next_arrays, axis=-1)
            next_state = processed_next_state
            done = temp_exp_mem[-1][4]
            if (t == 3):
                logger.debug("State shape: {}".format(processed_state.shape))
                logger.debug(
                    "Next state shape: {}".format(processed_next_state.shape))
            self.agent.memory.add_exp_processed(
                processed_state, action, reward,
                processed_next_state, next_state, done)
