from AI.State import State

from keras import layers, models
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
import numpy as np


class NetworkTracker:
    model: models
    target_model: models

    # pass in the environment which has input shape of the frame
    def __init__(self, environment, source: bool = False,
                 network_name: str = None):
        self.network_name = network_name
        if source:
            self.model = models.load_model(self.network_name)
        else:
            self.model = self.define_model(environment)
            self.model.save(self.network_name)
        self.target_model = self.model

    def define_model(self, env):
        # definition of the model is specific to the approach a person is taking
        raise NotImplementedError

    def get_q_values_for_one(self, state):
        # print("State : {}".format(state))
        # print("State shape: {}". format(state.shape))
        output_tensor = self.model.predict(state.reshape((1,) + state.shape))
        # the State class handles turning the state into an appropriate input
        # tensor for a NN so you don't have to change it everywhere
        return output_tensor[0]
        # want to convert the 2 dimensional output to 1 dimension to call argmax

    def get_max_q_value_index(self, state):
        return np.argmax(self.get_q_values_for_one(state))

    def get_q_values_for_batch(self, states):
        if states[0] is State:
            states = np.asarray(states)

        f = self.model.predict(states)
        return f

    def get_target_tensor(self, next_states):
        if next_states[0] is State:
            next_states = np.asarray(
                [i.to_input_tensor(individual=False) for i in next_states])
        output_tensor = self.target_model.predict(next_states)

        return output_tensor

    def fit(self, states_batch, targets_batch):
        self.model.fit(states_batch, targets_batch, verbose=1)

    def clone_policy(self):  # defining the target network
        self.model.save(self.network_name)
        self.target_model = models.load_model(self.network_name)

    def get_model_summary(self):
        return self.model.summary()
