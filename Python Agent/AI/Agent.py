import random


class Agent:

    def __init__(self, environment, network, run_only=False,
                 eps_decay_rate=0.9975, max_exp_rate=1.0, min_exp_rate=0.05):
        self.env = environment  # this should be the environment wrapper class

        if not run_only:
            # our network starts off with this exploration rate
            self.exp_rate = max_exp_rate
        else:
            self.exp_rate = 0.0

        # have at least 0.01 exploration rate at all times
        self.min_exp_rate = min_exp_rate

        # decay the exploration rate at this rate
        self.decay_rate = eps_decay_rate

        self.time_step = 0     # keeps track of time steps

        self.network = network

    def take_action(self, current_state):
        # Implement the epsilon greedy strategy

        # get a random number from 0 to 1 with linear distribution
        result = random.random()

        # if it falls over the explore rate, exploit
        if result > self.get_exp_rate():
            # Get the action with the maximum q-value
            action = self.env.get_action_at_index(
                self.network.get_max_q_value_index(current_state))  # exploit
            print("Network Algorithm")
        else:  # if it falls under the explore rate, explore
            # explore (generate a random action from the environment class)
            action = self.env.get_random_action()
            print("Random Action")

        # increment time step as well as update the decay rate
        self.increment_time_step()
        next_state, reward, done = self.env.step(
            action)  # finally, take the action and record the reward

        return current_state, self.env.action_space.index(action), reward, \
               next_state, done  # return an experience Tuple


    def reset_time_steps(self, i=0):
        self.timesteps = i

    def increment_time_step(self):
        self.time_step += 1

    def update_epsilon(self):
        if self.exp_rate > self.min_exp_rate:
            self.exp_rate = self.exp_rate * self.decay_rate
        else:
            self.exp_rate = self.min_exp_rate

    def get_exp_rate(self):
        return self.exp_rate
