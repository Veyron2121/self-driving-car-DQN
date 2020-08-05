import json
import random
from tkinter import Image
from typing import Dict

import numpy as np
import zmq
from PIL import Image

from ActionSpace import Acc, Steer
from AI.Frame import Frame
from AI.State import State
from AI.DataBuffer import DataBuffer


class RNNEnvironmentWrapper:

    def __init__(self):

        print("Waiting to connect to Simulator...")

        context = zmq.Context()
        # noinspection PyUnresolvedReferences
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        print("Connected to simulator!")

        print("Getting Image Path")
        self.image_path = self.socket.recv().decode('ascii')
        # self.socket.send("Path Received".encode('ascii'))

        print("Image path received: " + str(self.image_path))

        # initial_state_info = self.get_next_state()
        # speed, angle, distance = self.get_game_stats(initial_state_info)

        # initialising frame buffer
        self.buffer_size = 50  # could change this

        self.current_buffer = DataBuffer(size=self.buffer_size)
        self.current_state = None

        _ = self.reset()

        self.time_steps = 0

        self.done = False

        self.max_time_steps_per_episode = 500  # change this based on the environment

        self.action_space = [(Acc.BRAKE, Steer.TURN_RIGHT),
                             (Acc.BRAKE, Steer.DO_NOTHING),
                             (Acc.BRAKE, Steer.TURN_LEFT),
                             (Acc.DO_NOTHING, Steer.TURN_RIGHT),
                             (Acc.DO_NOTHING, Steer.DO_NOTHING),
                             (Acc.DO_NOTHING, Steer.TURN_LEFT),
                             (Acc.ACCELERATE, Steer.TURN_RIGHT),
                             (Acc.ACCELERATE, Steer.DO_NOTHING),
                             (Acc.ACCELERATE, Steer.TURN_LEFT)]

        self.current_frame = None

    def get_input_shape(self):
        """
        returns the input shape for the input layer of the network
        """
        return self.current_buffer.get_input_shape()

    # def get_state_shape(self):
    #     """
    #     not to be confused with the input shape. This is the shape of individual state (the shape of an individual processed shape of the environment)
    #     """
    #     return self.current_state.get_shape()

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

    def get_action_at_index(self, index):
        return self.action_space[index]

    def reset(self):
        """
        resets the environment. self.done denotes whether the episode is done i.e. the car has crashed or we have stopped it
        """
        self.done = False
        self.time_steps = 0

        # reset the sim and get the initial frame from it in the folder

        path = self.image_path + "/Screenshots/PerspectiveSegment_1.png"

        info = self.step("reset")
        current_state, _, self.done = info
        self.current_state = current_state[0]
        # print("Reset Current State: {} {}".format(current_state, self.current_state))

        return current_state

    def step(self, action):
        """
        does the action and returns the reward for that action along with the next state

        This function may get complicated as it has to interact with the car simulator throught sockets.
        """
        self.time_steps += 1

        reward = 0
        if not self.is_done():
            action_string = str(action[0]) + ',' + str(action[1])
            # print(action)
            if not action == "reset":
                reward = self.get_basic_reward(self.current_state[0],
                                               self.current_state[2],
                                               action)

            self.socket.send(action_string.encode('ascii'))

            data = self.get_next_state()
            # print("Received request: %s" % message)
            speed, angle, distance = self.get_game_stats(data)
            state = State((speed, angle, distance))
            if self.current_buffer.reinitialize_next:
                # only happens if there aren't <timesteps> yet available
                # reinitialize_buffer replicates the input state <size> times into the buffer
                self.current_buffer.reinitialize_buffer(state)
            self.current_buffer.assign_to_buffer(state)

            self.done = data["is_done"]

            buffer_tensor = self.current_buffer.get_input_tensor()
            # A buffer is a collection of consecutive frames that we feed to the NN.
            # These frames are already processed.

            # the current state consists of all the information from the
            # environment
            self.current_state = (angle, distance, speed)

            if self.done:
                self.current_buffer.set_reinitialize_next(True)

            # this returns the state of the environment after the action has
            # been completed, the reward for the action and if the
            # episode ended.
            return buffer_tensor, reward, self.done
        else:
            return None

    def get_next_state(self) -> Dict:
        # print("Getting next state")
        message = self.socket.recv()
        # print("State Received: " + message.decode('ascii'))
        # unpack JSON
        return json.loads(message)

    def get_basic_reward(self, angle, speed, action):
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

    def get_frame(self, path: str) -> Frame:
        """
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
        return self.current_buffer

    def get_game_stats(self, data):
        """
        returns a tuple of angle, distance from checkpoint and speed from the sim. Again requires comms with simulator.
        """
        return data["velocity"], data["angle_from_road"], \
               data["distance_from_road"]
