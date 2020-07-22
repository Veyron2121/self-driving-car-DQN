import numpy as np
from typing import List, Tuple

class State:
    """Keeps track of the current state"""

    state_data: np.array

    def __init__(self, state_data: Tuple[float]):
        self.state_data = np.asarray(state_data)

    def process_state(self):
        pass

    def get_batch_tensor(self) -> np.array:
        holder = np.asarray(self.state_data)
        holder.reshape((1,) + holder.shape)
        return holder

    def get_individual_tensor(self) -> np.array:
        """Returns the individual tensor"""
        return np.asarray(self.state_data)

    def get_shape(self) -> Tuple[int]:
        """Returns the shape of the state data"""
        return self.state_data.shape

    def display(self) -> None:
        """Prints the states"""
        print(self.state_data)
