from AI.NetworkTracker import NetworkTracker
from keras import models, layers


class NetworkTrackerRNN(NetworkTracker):

    # pass in the environment which has input shape of the frame
    def __init__(self, environment, source: bool = False,
                 network_name: str = 'DrivePolicyRNN.h5'):
        super.__init__(environment, source, network_name)

    def define_model(self, env):
        model = models.Sequential()

        # Add a LSTM layer with 32 internal units.
        model.add(layers.LSTM(32, activation='tanh',
                              input_shape=env.get_input_shape()))

        # Add a Dense layer with <action space> units.
        model.add(layers.Dense(env.get_num_action_space(), activation='linear'))

        model.compile(optimizer='adam', loss='mse')

        return model
