import os
from pathlib import Path

from FrameBuffer import FrameBuffer

class EnvironmentWrapper:

    def __init__(self):

        # TODO: Change socket behaviour here - Varun
        # initialise comms with the simulator here
        sock = socket.socket(socket.AF_INET,
                             socket.SOCK_STREAM)  # initialise the socket
        # connect to localhost, port 2345
        sock.bind(("127.0.0.1", 4444))
        sock.listen(1)
        print("Waiting to connect to Simulator...")
        self.clientsocket, _ = sock.accept()  # connect to simulator [BLOCKING]
        print("Connected to simulator!")
        # =========================================

        # initialising frame buffer
        self.buffer_size = 4.  # could change this

        self.current_buff = FrameBuffer(
            size=self.buffer_size)  # this is the FrameBuffer that keeps track of the latest frames.
        # initialising the frame buffer by just giving it multiple copies of the same starting frame
        # =========================================

        _ = self.reset()

        self.prev_dist = 0

        self.time_steps = 0

        self.done = False

        self.max_time_steps_per_episode = 500  # change this based on the enviorment

    def get_input_shape(self):
        """
        returns the input shape for the input layer of the network
        """
        return self.buffer.get_input_shape()

    def get_state_shape(self):
        """
        not to be confused with the input shape. This is the shape of individual state (the shape of an individual processed shape of the environment)
        """
        return self.current_state.get_shape()

    def get_random_action(self):
        """
        returns a random action to be taken. [steering, acceleration, brake]
        """
        raise NotImplementedError

    def get_num_action_space(self):
        """
        returns the number of permuations of valide actions. For Eg. [steer left, accelerate and no brake] is ONE action
        [steer right, accelerate and brake] is invalid as we cannot accelerate and brake at the same time.
        there are 9 possible actions I think?
        """
        # TODO: Maybe make this more general?
        return 9

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
            self.current_buff.assign_to_buffer(self.current_frame)

        return self.current_buff

    def step(self, action):
        """
        does the action and returns the reward for that action along with the next state

        This function may get complicated as it has to interact with the car simulator throught sockets.
        """
        self.time_steps += 1

        if not self.is_done():
            # if the episode has not ended
            # =======================
            # in this section of the code, we wait for the simulator for to give us an action request
            # then we give it an action through the TCP/IP protocol
            msg = self.receive_message()
            if msg == 'requesting action':
                # send the action
                self.send_message('action'.encode())
                # now wait for the action to be completed
                action_done = self.receive_message()

                if action_done == 'action done':
                    # the 4 images will be stored as scr{i}.png
                    for i in range(1, 4):
                        # each Frame object is then assigned to the FrameBuffer class in chronological order
                        path = Path(os.getcwd()).parent / 'scr.{0}.png'.format(
                            i)
                        f = self.get_frame(path)
                        self.current_buff.assign_to_buffer(f)

                        angle, distance, speed = self.get_game_stats()

                    # still need to figure out how to send over distance, speed angle data

                    dist_delta = self.prev_dist - distance

                    if abs(dist_delta) > 30:
                        dist_delta = 5  # if there's too big a negative jump in the distance, the car has passed a checkpoint.
                        # so, don't penalise it for that.

                    # calculate reward based on the game stats
                    reward = (dist_delta * 0.7) + (speed * 0.3) - (angle * 0.1)

            # this buffer tensor is what we will input to the NN in the next time step
            # we record this as the next observation.
            buffer_tensor = self.current_buff.to_input_tensor()
            # A buffer is a collection of consecutive frames that we feed to the NN. These frames are already processed.

            # this returns the state of the environment after the action has been completed, the reward for the action and if the episode ended.
            return buffer_tensor, reward, self.done
        else:
            return None

    def send_message(self, string):
        try:
            self.clientsocket.sendall(string.encode())
        except:
            print("Socket Exception while sending data to simulator")

    def receive_message(self):
        data = None
        try:
            data = self.clientsocket.recv(128).decode()
        except:
            print("Socket Exception while recieving data from simulator")

        return data

    def is_done(self):
        """
        returns if the episode is finished
        """
        raise NotImplementedError

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

    def get_current_state():
        """
        get the last n frames from the simulator (they might be stored in a folder by the simulator)
        and store them in a buffer and return them
        """
        return self.current_buff

    def get_game_stats(self):
        """
        returns a tuple of angle, distance from checkpoint and speed from the sim. Again requires comms with simulator.
        """
        self.send_message("requesting info")
        string = self.receive_message()

        # TODO: SOBI parse this JSON string which will contain game information and return

