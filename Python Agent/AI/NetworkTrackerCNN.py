from AI.NetworkTracker import NetworkTracker
from keras import layers, models


class NetworkTrackerCNN(NetworkTracker):
    model: models
    target_model: models

    # pass in the environment which has input shape of the frame
    def __init__(self, environment, source: bool = False,
                 network_name: str = 'DrivePolicyCNN.h5'):
        super.__init__(environment, source, network_name)

    def define_model(self, env):
        model = models.Sequential()

        model.add(layers.Conv2D(filters=10, kernel_size=(3, 3),
                                activation='relu',
                                input_shape=env.get_input_shape()))
        # first layer takes input shape from the environment
        print(env.get_input_shape())

        model.add(layers.MaxPool2D((3, 3)))

        model.add(layers.Conv2D(filters=20, kernel_size=(3, 3), strides=2,
                                activation='relu'))

        model.add(layers.MaxPool2D(3, 3))

        model.add(layers.Flatten())

        model.add(layers.Dense(16, activation='sigmoid'))

        model.add(layers.Dense(16, activation='relu'))

        model.add(layers.Dense(env.get_num_action_space(), activation='linear'))

        model.compile(optimizer=Adam(lr=0.0001), loss='mse')

        return model
