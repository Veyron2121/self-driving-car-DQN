from ActionSpace import Acc, Steer
from typing import Tuple

from .Controller import Controller


class HandCraftedController(Controller):
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

        acc = Acc.DO_NOTHING
        steer = Steer.DO_NOTHING

        if velocity > self.v_brake:
            acc = Acc.BRAKE
        if velocity < self.v_max:
            acc = Acc.ACCELERATE
        if angle > self.phi_threshold:
            steer = Steer.TURN_LEFT
        if angle < -self.phi_threshold:
            steer = Steer.TURN_RIGHT

        return acc, steer
