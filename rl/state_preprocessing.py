from rl.util import *

def action_sel_processing_stack_states(state, previous_state):
        return np.concatenate([previous_state, state])

def action_sel_processing_diff_states(state, previous_state):
    return state - previous_state    

def action_sel_processing_atari_states(state, previous_state, 
                                      pre_previous_state, pre_pre_previous_state):
    arrays = (process_image_atari(state),
              process_image_atari(previous_state),
              process_image_atari(pre_previous_state),
              process_image_atari(pre_pre_previous_state))
    return np.stack(arrays, axis=-1)

def run_state_processing_none(agent, temp_exp_mem, t):
    # No state processing
    state = temp_exp_mem[-1][0]
    action = temp_exp_mem[-1][1]
    reward = temp_exp_mem[-1][2]
    next_state = temp_exp_mem[-1][3]
    done = temp_exp_mem[-1][4]
    agent.memory.add_exp_processed(state, action, reward, 
                                   next_state, next_state, done)

def run_state_processing_stack_states(agent, temp_exp_mem, t):
    # Concatenates previous + current states
    if (t >=1):    
        processed_state = np.concatenate([temp_exp_mem[-2][0], temp_exp_mem[-1][0]]) 
        action = temp_exp_mem[-1][1]
        reward = temp_exp_mem[-1][2]
        processed_next_state = np.concatenate([temp_exp_mem[-1][0], temp_exp_mem[-1][3]])
        next_state = processed_next_state
        done = temp_exp_mem[-1][4]
        if (t == 1):
            print("State shape: {}".format(processed_state.shape))
            print("Next state shape: {}".format(processed_next_state.shape))
        agent.memory.add_exp_processed(processed_state, action, reward, 
                                       processed_next_state, next_state, done)

def run_state_processing_diff_states(agent, temp_exp_mem, t):
    # Change in state params, curr_state - last_state
    if (t >= 1):
        processed_state = temp_exp_mem[-1][0] - temp_exp_mem[-2][0]
        action = temp_exp_mem[-1][1]
        reward = temp_exp_mem[-1][2]
        processed_next_state = temp_exp_mem[-1][3] - temp_exp_mem[-1][0]
        next_state = processed_next_state
        done = temp_exp_mem[-1][4]
        if (t == 1):
            print("State shape: {}".format(processed_state.shape))
            print("Next state shape: {}".format(processed_next_state.shape))
        agent.memory.add_exp_processed(processed_state, action, reward, 
                                       processed_next_state, next_state, done)

def run_state_processing_atari(agent, temp_exp_mem, t):
    # Convert images to greyscale, downsize, crop, then stack 4 states
    # NOTE: Image order is cols * rows * channels to match openai gym format
    # Input to model is rows * cols * channels (== states)
   if (t>=3):
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
            print("State shape: {}".format(processed_state.shape))
            print("Next state shape: {}".format(processed_next_state.shape))
        agent.memory.add_exp_processed(processed_state, action, reward, 
                                       processed_next_state, next_state, done)