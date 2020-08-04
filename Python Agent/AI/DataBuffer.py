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

    def __init__(self, size: int = 1):
        self.buffer = []
        self.size = size

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

    def get_input_shape(self) -> Tuple[int]:
        """
        Returns the shape of the buffer as a tuple
        :return: the shape of the buffer
        """
        return np.asarray(self.buffer).shape

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
