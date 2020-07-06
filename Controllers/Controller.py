from abc import ABC, abstractmethod
from ActionSpace import Acc, Steer
from typing import Tuple


class Controller(ABC):

    v_brake: float
    v_max: float
    phi_threshold: float

    def __init__(self, v_brake: float, v_max: float, phi_threshold: float):
        self.v_brake = v_brake
        self.v_max = v_max
        self.phi_threshold = phi_threshold

    @abstractmethod
    def get_car_policy(self, velocity: float, angle: float, distance: float,
                       image: str) -> Tuple[Acc, Steer]:
        """
        :param velocity: the current velocity of the car
        :param angle: the angle between the car's heading to the next checkpoint
        :param distance: the distance from the car to the next checkpoint
        :param image: the perspective of the car from the camera
        :return: a tuple containing the next acceleration and steering actions
        """
        pass
