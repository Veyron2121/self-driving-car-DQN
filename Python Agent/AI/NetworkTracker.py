from AI.State import State

network_name = 'NewPolicy.h5'
from keras import layers, models
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
import numpy as np

class NetworkTracker:


    model: models
    target_model: models

    def __init__(self, environment, source: bool = False):  # pass in the environment which has input shape of the frame
        if source:
            self.model = models.load_model(network_name)
        else:
            self.model = self.define_model(environment)
            self.model.save(network_name)
        self.target_model = self.model

    def define_model(self, env): #defination of the model, its specific to the approach a person is taking.
        model = models.Sequential()
        model.add(
            layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu',
                          input_shape=env.get_input_shape()))

        model.add(layers.MaxPool2D((3, 3)))

        model.add(
            layers.Conv2D(filters=20, kernal_size=(2, 2), activation='relu'))

        model.add(layers.MaxPool2D(3, 3))

        model.add(layers.Flatten())

        model.add(layers.Dense(16, activation='softmax'))

        model.add(layers.Dense(16, activation='relu'))

        model.add(layers.Dense(env.get_num_action_space(), activation='linear'))

        model.compile(optimizer='adam',
                      loss='mse')
        return model

    def get_q_values_for_one(self, state):

        output_tensor = self.model.predict(state.reshape(
            (1,) + state.shape))  # the State class handles turning the state
        # into an appropriate input tensor for a NN
        # so you don't have to change it everywhere
        return output_tensor[
            0]  # you want to convert the 2 dimensional output to 1 dimension to call argmax

    def get_max_q_value_index(self, state): #self explanatory 
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

    def clone_policy(self): #defining the target network
        self.model.save(network_name)
        self.target_model = models.load_model(network_name)

    def get_model_summary(self):
        return self.model.summary()
