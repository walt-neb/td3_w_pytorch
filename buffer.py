#filename: buffer.py


import numpy as np
import pickle
# (state, action, reward, next_state, done)

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size

        # Check the type of the 'state'
        if isinstance(state, np.ndarray):
            self.state_memory[index] = state
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
            self.state_memory[index] = state[0]  # only ndarray part of tuple is used
        else:
            raise TypeError(
                f'Unsupported state type: {type(state)}, expecting numpy array or a tuple consisting of numpy array and a dict.')

        #self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        if max_mem == 0:
            max_mem = 1
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, next_state, dones


def save_replaybuffer(replay_buffer, replaybuffer_file):
    try:
        with open(replaybuffer_file, 'wb') as f:
            pickle.dump(replay_buffer, f)
            print(f'Replay buffer saved to {replaybuffer_file}')
    except Exception as e:
        print('Failed to save replay buffer {}: {}'.format(replaybuffer_file, e))


def load_replaybuffer(replaybuffer_file):
    try:
        with open(replaybuffer_file, 'rb') as f:
            replay_buffer = pickle.load(f)
            print(f'Replay buffer loaded from {replaybuffer_file}')
            return replay_buffer
    except Exception as e:
        print('Failed to load replay buffer from {}'.format(replaybuffer_file))
        print(e)
        return None  # Return None or raise the exception again if you want to handle it outside


