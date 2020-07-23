import json
import os
import random
from pathlib import Path
from tkinter import Image

import numpy as np
import zmq
import cv2

from ActionSpace import Acc, Steer
from .Frame import Frame
from .FrameBuffer import FrameBuffer

class EnvironmentWrapper:

    def __init__(self):

        # TODO: Change socket behaviour here - Varun
        print("Waiting to connect to Simulator...")

        context = zmq.Context()
        # noinspection PyUnresolvedReferences
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        print("Connected to simulator!")


        # initialising frame buffer
        self.buffer_size = 4  # could change this

        self.current_buffer = FrameBuffer(
            size=self.buffer_size)  # this is the FrameBuffer that keeps track of the latest frames.
        # initialising the frame buffer by just giving it multiple copies of the same starting frame
        # =========================================

        _ = self.reset()

        self.prev_dist = 0

        self.time_steps = 0

        self.done = False

        self.max_time_steps_per_episode = 500  # change this based on the environment

        self.action_space = [Acc.BRAKE, Steer.TURN_RIGHT,
                             Acc.BRAKE, Steer.DO_NOTHING,
                             Acc.BRAKE, Steer.TURN_RIGHT.
                             Acc.DO_NOTHING, Steer.TURN_RIGHT,
                             Acc.DO_NOTHING, Steer.DO_NOTHING,
                             Acc.DO_NOTHING, Steer.TURN_RIGHT,
                             Acc.ACCELERATE, Steer.TURN_RIGHT,
                             Acc.ACCELERATE, Steer.DO_NOTHING,
                             Acc.ACCELERATE, Steer.TURN_RIGHT]

        self.current_frame = None

    def get_input_shape(self):
        """
        returns the input shape for the input layer of the network
        """
        return self.current_buffer.get_input_shape()

    def get_state_shape(self):
        """
        not to be confused with the input shape. This is the shape of individual state (the shape of an individual processed shape of the environment)
        """
        return self.current_state.get_shape()

    def get_random_action(self):
        """
        returns a random action to be taken. [steering, acceleration, brake]
        """
        return random.choice(self.action_space)

    def get_num_action_space(self) -> int:
        """
        returns the number of permuations of valid actions. For Eg. [steer left, accelerate and no brake] is ONE action
        [steer right, accelerate and brake] is invalid as we cannot accelerate and brake at the same time.
        there are 9 possible actions I think?
        """

        return len(self.action_space)

    def reset(self):
        """
        resets the environment. self.done denotes whether the episode is done i.e. the car has crashed or we have stopped it
        """
        self.done = False
        self.time_steps = 0

        # reset the sim and get the initial frame from it in the folder

        path = Path(os.getcwd()).parent / 'scr1.png'

        self.current_frame = self.get_frame(path)

        for _ in range(int(self.buffer_size)):
            self.current_buffer.assign_to_buffer(self.current_frame)

        return self.current_buffer

    def step(self, action):
        """
        does the action and returns the reward for that action along with the next state

        This function may get complicated as it has to interact with the car simulator throught sockets.
        """
        self.time_steps += 1

        if not self.is_done():
            action_string = str(action[0]) + ',' + str(action[1])

            self.socket.send(action_string.encode('ascii'))

            message = self.socket.recv()
            # unpack JSON
            data = json.loads(message)
            # print("Received request: %s" % message)
            speed, angle, distance = self.get_game_stats(data)

            f = self.get_frame(data["image_path"])
            self.current_buffer.assign_to_buffer(f)

            reward = self.get_basic_reward(speed, angle, action)
            self.done = data["is_done"]

            # this buffer tensor is what we will input to the NN in the next time step
            # we record this as the next observation.
            buffer_tensor = self.current_buffer.get_input_tensor()
            # A buffer is a collection of consecutive frames that we feed to the NN. These frames are already processed.

            # this returns the state of the environment after the action has been completed, the reward for the action and if the episode ended.
            return buffer_tensor, reward, self.done
        else:
            return None

    def get_basic_reward(self, speed, angle, action):
        v_min = 5
        v_brake = 20
        v_max = 18
        phi_threshold = 0.05
        reward = 0
        acc = action[0]
        steer = action[1]

        if speed > v_brake and acc == Acc.BRAKE:
            reward += 1
        if speed > v_brake and acc == Acc.ACCELERATE:
            reward -= 1
        if speed < v_max and acc == Acc.ACCELERATE:
            reward += 1
        if speed < v_max and acc == Acc.BRAKE:
            reward -= 1

        if angle < -phi_threshold and steer == Steer.TURN_RIGHT:
            reward += 1
        if angle < -phi_threshold and steer == Steer.TURN_LEFT:
            reward -= 1
        if angle > phi_threshold and steer == Steer.TURN_LEFT:
            reward += 1
        if angle > phi_threshold and steer == Steer.TURN_RIGHT:
            reward -= 1

        if speed < v_min and acc != Acc.ACCELERATE:
            reward -= 1

        return reward

    def is_done(self):
        """
        returns if the episode is finished
        """
        return self.done

    def close(self):
        """
        in case we need to 'close' the environment
        """
        raise NotImplementedError

    def get_frame(self, path: str) -> Frame:
        """
        communicates with the sim to get the latest state/frame.
        returns a Frame object
        Get image from path then convert to np array then make a frame object
        """
        image = Image.open(path, 'r')
        image.load()
        np_data = np.asarray(image, dtype="float32")
        return Frame(np_data)

    def get_current_state(self):
        """
        get the last n frames from the simulator (they might be stored in a folder by the simulator)
        and store them in a buffer and return them
        """
        return self.current_buff

    def get_game_stats(self, data):
        """
        returns a tuple of angle, distance from checkpoint and speed from the sim. Again requires comms with simulator.
        """
        return data["velocity"], data["angle_from_road"], \
               data["distance_from_road"]
