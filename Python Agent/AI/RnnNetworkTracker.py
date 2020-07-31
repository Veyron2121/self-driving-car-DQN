from AI import NetworkTracker
from keras import models, layers
from keras.layers import LeakyReLU

network_name = 'TalesDrivePolicyRNN.h5'

class RnnNetworkTracker(NetworkTracker):
    def __init__(self, environment,
                 source: bool = False):  # pass in the environment which has input shape of the frame
        super().__init__(environment, source)

    def define_model(self, env):
        model = models.Sequential()

        # Add a LSTM layer with 32 internal units.
        model.add(layers.LSTM(32, activation='tanh',
                              input_shape=env.get_input_shape()))

        # Add a Dense layer with <action space> units.
        model.add(layers.Dense(env.get_num_action_space(), activation='linear'))

        model.compile(optimizer='adam', loss='mse')

        return model
