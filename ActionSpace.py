from enum import Enum


class Acc(Enum):
    BRAKE = 1
    DO_NOTHING = 2
    ACCELERATE = 3


class Steer(Enum):
    TURN_LEFT = 1
    DO_NOTHING = 2
    TURN_RIGHT = 3
