import numpy as np
from tensorflow.python.keras import Model

from AI.NetworkTracker import NetworkTracker
from keras import models, layers
from keras.layers import Dense, LeakyReLU, concatenate

from AI.State import State

network_name = 'VarunDrivePolicyHybrid.h5'
class HybridNetworkTracker(NetworkTracker):

    def __init__(self, environment,
                 source: bool = False):  # pass in the environment which has input shape of the frame
        if source:
            self.model = models.load_model(network_name)
        else:
            self.model = self.define_model(environment)
            self.model.save(network_name)
        self.target_model = self.model

    def define_model(self, env):
        leaky_relu_alpha = 0.3
        cnn = models.Sequential()
        cnn.add(
            layers.Conv2D(filters=1, kernel_size=(11, 11),
                          input_shape=env.get_input_shape()[0]))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(9, 9)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(7, 7)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(5, 5)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(3, 3)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dense(4))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        # cnn.compile(optimizer='adam', loss='mse')
        # cnn.add(layers.Dense(9))
        # cnn.add(LeakyReLU(alpha=leaky_relu_alpha))

        rnn = models.Sequential()

        # Add a LSTM layer with 32 internal units.
        rnn.add(layers.LSTM(32, activation='tanh',
                            input_shape=env.get_input_shape()[1]))  # env.get_input_shape
        # print("Input Shape: {}".format(env.get_input_shape()))
        # rnn.compile(optimizer='adam', loss='mse')
        # Add a Dense layer with <action space> units.
        # rnn.add(layers.Dense(9, activation='linear'))

        concatenated = concatenate([cnn.output, rnn.output], axis=-1)
        concatenated = Dense(18, activation='sigmoid')(concatenated)
        concatenated = Dense(9, activation='sigmoid')(concatenated)

        hybrid = Model(inputs=[cnn.input, rnn.input], outputs=concatenated)
        hybrid.compile(optimizer='adam', loss='mse')

        return hybrid

    # TODO: Convert these things to account for two different input in a concat

    def get_q_values_for_one(self, state):
        # print("State : {}".format(state))
        # print("State shape: {}". format(state.shape))
        output_tensor = self.model.predict([state[0].reshape(
            (1,) + state[0].shape), state[1].reshape(
            (1,) + state[1].shape)])  # the State class handles turning the state
        # into an appropriate input tensor for a NN
        # so you don't have to change it everywhere
        return output_tensor[0]
        # you want to convert the 2 dimensional output to 1 dimension to call argmax

    def get_max_q_value_index(self, state): # self explanatory
        return np.argmax(self.get_q_values_for_one(state))

    def get_q_values_for_batch(self, states):
        # if states[0][0] is State:
        #     states = np.asarray(states)
        frame_information = np.asarray([i[0] for i in states])
        # print(frame_information.shape)
        data_information = np.asarray([i[1] for i in states])
        # print(data_information.shape)
        f = self.model.predict([frame_information, data_information])
        return f

    def get_target_tensor(self, next_states):
        frame_information = np.asarray([i[0] for i in next_states])
        # print(frame_information.shape)
        data_information = np.asarray([i[1] for i in next_states])
        if next_states[0] is State:
            next_states = np.asarray(
                [i.to_input_tensor(individual=False) for i in next_states])
        output_tensor = self.target_model.predict([frame_information,
                                                   data_information])

        return output_tensor

    def fit(self, states_batch, targets_batch):

        frame_information = np.asarray([i[0] for i in states_batch])
        data_information = np.asarray([i[1] for i in states_batch])
        print("Fitting Model with shape: {}, {}".format(frame_information.shape,
                                                        data_information.shape))
        self.model.fit([frame_information, data_information], targets_batch, verbose=1)

