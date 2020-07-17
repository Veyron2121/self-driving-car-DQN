import numpy as np


class DataBuffer:
    """
    Keeps track of n latest states
    """

    def __init__(self, size: int):
        self.buffer = []
        self.size = size

    def get_input_tensor(self, in_batch=True):
        arr = np.array(self.buffer)
        if size == 1 or in_batch:
            return arr
        else:
            return arr.reshape((1,) + arr.shape)

    def get_input_shape(self):
        return np.asarray(self.buffer).shape

    def assign_to_buffer(self, state):
        if isinstance(state, State):
            state = state.get_individual_tensor()
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(state)
