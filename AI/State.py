class State:
    def __init__(self, state_data):
        self.state_data = np.asarray(state_data)

    def process_state(self):
        pass

    def get_batch_tensor(self):
        holder = np.asarray(self.state_data)
        holder.reshape((1,) + holder.shape)
        return holder

    def get_individual_tensor(self):
        return np.asarray(self.state_data)

    def get_shape(self):
        return self.state_data.shape

    def display(self):
        print(self.state_data)
