
import random
class Memory:
    def __init__(self, size):
        self.replay = []
        self.limit = size
        self.exp_count = 0

    def push(self, experience):
        self.exp_count += 1

        if self.exp_count < self.limit:
            self.replay.append(experience)  #append to experiences
        else:
            self.replay[self.exp_count%len(self.replay)] = experience  #wrap around if the memory capacity is reached
        assert len(self.replay) <= self.limit

    def is_usable(self, batch_size):
        return len(self.replay) >= batch_size

    def reset_replay_memory(self):
        self.exp_count = 0
        self.replay = []

    def sample(self, batch_size):
        return random.sample(self.replay, batch_size)
