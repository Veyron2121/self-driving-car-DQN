import numpy as np
from .State import State
from typing import List, Tuple


class DataBuffer:
    """
    Keeps track of <size> latest states
    """
    # This list keeps track of the <size> latest states that the agent has
    # encountered
    buffer: List[State]
    # The number of states we're keeping track of
    size: int
    # Whether to reinitialize the buffer at the next iteration
    reinitialize_next: bool

    def __init__(self, size: int = 1):
        self.buffer = []
        self.size = size
        self.reinitialize_next = True

    def get_input_tensor(self, in_batch: bool = True) -> np.array:
        """

        :param in_batch:
        :return:
        """
        arr = np.array(self.buffer)
        if self.size == 1 or in_batch:
            return arr
        else:
            return arr.reshape((1,) + arr.shape)

    def get_input_shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the buffer as a tuple
        :return: the shape of the buffer
        """
        return self.size, 3

    def assign_to_buffer(self, state: State) -> None:
        """
        Adds <state> to the buffer
        :param state: the latest state observed by the agent
        :return: None
        """
        if isinstance(state, State):
            state = state.get_individual_tensor()
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(state)

    def reinitialize_buffer(self, state: State):
        for _ in range(self.size):
            self.assign_to_buffer(state)
        self.set_reinitialize_next(False)

    def set_reinitialize_next(self, reinitialize: bool):
        self.reinitialize_next = reinitialize
