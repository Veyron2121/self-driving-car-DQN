import numpy as np

from AI.DataBuffer import DataBuffer
from AI.State import State


class FrameBuffer(DataBuffer):
    "Keeps track of <size> latest frames"

    size: int

    def __init__(self, size = 3):
        DataBuffer.__init__(self, size=size)

    def assign_to_buffer(self, state: State) -> None:
        """Adds frame to buffer"""
        if state is State:
            state = state.get_individual_tensor()

        # if buffer not initialised
        if not self.buffer:
            self.buffer = np.array([state])
            return

        # if buffer size reached, delete the oldest frame
        if self.buffer.shape[-1] >= self.size:
            self.buffer = self.buffer[: ,: ,1:]

            # The below line stacks the new frame in the buffer with shape (height, width, buffer_size)
            #         self.buffer = np.concatenate((self.buffer, state), axis=2)
            self.buffer = np.concatenate((self.buffer, state), axis=0)
