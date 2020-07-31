from AI import NetworkTracker
from keras import models, layers
from keras.layers import LeakyReLU

class HybridNetworkTracker(NetworkTracker):
    def __init__(self, environment,
                 source: bool = False):  # pass in the environment which has input shape of the frame
        super().__init__(environment, source)

    def define_model(self, env):
        leaky_relu_alpha = 0.3
        cnn = models.Sequential()
        cnn.add(
            layers.Conv2D(filters=1, kernel_size=(11, 11),
                          input_shape=env.get_input_shape()))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(9, 9)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(7, 7)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(5, 5)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Conv2D(filters=1, kernel_size=(3, 3)))
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Dense(env.get_num_action_space())
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.add(layers.Dense(env.get_num_action_space())
        cnn.add(LeakyReLU(alpha=leaky_relu_alpha))
        cnn.compile(optimizer='adam', loss='mse')

        rnn = models.Sequential()
        # Add a LSTM layer with 32 internal units.
        rnn.add(layers.LSTM(32))
        # Add a Dense layer with 9 units.
        rnn.add(layers.Dense(9))
        rnn.compile(optimizer='adam', loss='mse')

