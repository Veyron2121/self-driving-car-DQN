import random
from typing import List, Tuple, Any


class Memory:
    """Stores replay experiences in a tuple"""

    replay: List[Tuple]
    size: int
    exp_count: int

    def __init__(self, size):
        self.replay = []
        self.limit = size
        self.exp_count = 0

    def push(self, experience) -> None:
        """Adds an experience to memory"""
        self.exp_count += 1

        if self.exp_count < self.limit:
            self.replay.append(experience)  # append to experiences
        else:
            self.replay[self.exp_count % len(self.replay)] = experience  # wrap around if the memory capacity is reached
        assert len(self.replay) <= self.limit

    def is_usable(self, batch_size: int) -> bool:
        """Determines if memory is accessible with this batch size"""
        return len(self.replay) >= batch_size

    def reset_replay_memory(self) -> None:
        """Resets memory"""
        self.exp_count = 0
        self.replay = []

    def sample(self, batch_size: int) -> List[Tuple]:
        """Randomly samples an experience from memory"""
        return random.sample(self.replay, batch_size)
