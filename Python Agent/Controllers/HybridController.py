from ActionSpace import Acc, Steer
from typing import Tuple

from .Controller import Controller

from keras import models


class HybridController(Controller):

    v_brake: float
    v_max: float
    phi_threshold: float

    def __init__(self, v_brake: float, v_max: float, phi_threshold: float):
        super().__init__(v_brake, v_max, phi_threshold)

    def get_car_policy(self, velocity: float, angle: float, distance: float,
                       image: str) -> Tuple[Acc, Steer]:
        """
        :param velocity: the current velocity of the car
        :param angle: the angle between the car's heading to the next checkpoint
        :param distance: the distance from the car to the next checkpoint
        :param image: the perspective of the car from the camera
        :return: a tuple containing the next acceleration and steering actions
        """
        cnn = models.Sequential()
        cnn.add()
        rnn = models.Sequential()

        return Acc.ACCELERATE, Steer.TURN_RIGHT
