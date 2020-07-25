import numpy as np

from AI.DataBuffer import DataBuffer
from AI.State import State


class FrameBuffer(DataBuffer):
    "Keeps track of <size> latest frames"

    size: int

    def __init__(self, size = 3):
        DataBuffer.__init__(self, size=size)

    def get_input_shape(self):
        return self.get_input_tensor().shape

    def get_input_tensor(self, in_batch=True):
        temp = np.array(self.buffer)
        return temp.transpose((1, 2, 0))
        # return temp

    def assign_to_buffer(self, state: State) -> None:
        """Adds frame to buffer"""
        if isinstance(state, State):
            state = state.get_individual_tensor()

        # if buffer not initialised
        if len(self.buffer) == 0:
            self.buffer = [state]
            return

        # if buffer size reached, delete the oldest frame
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)

        self.buffer.append(state)
