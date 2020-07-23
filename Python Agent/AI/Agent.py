import random


class Agent:

    def __init__(self, environment, network, run_only=False, eps_decay_rate=0.9975,max_exp_rate=1.0, min_exp_rate=0.05):
        self.env = environment # this should be the environment wrapper class

        if not run_only:
            self.exp_rate = max_exp_rate     # our network starts off with this exploration rate
        else:
            self.exp_rate = 0.0

        self.min_exp_rate = min_exp_rate  # have at least 0.01 exploration rate at all times

        self.decay_rate = eps_decay_rate   # decay the exploration rate at this rate

        self.time_step = 0     # keeps track of time steps

        self.network = network

    def take_action(self, current_state):
        # Implement the epsilon greedy strategy
        result = random.random()                      # get a random number from 0 to 1 with linear distribution
        explored = False
        if result > self.get_exp_rate():              # if it falls over the explore rate, exploit
            action = self.network.get_max_q_value_index(current_state)  # exploit

        else:                                         # if it falls under the explore rate, explore
            action = self.env.get_random_action()          # explore (generate a random action from the environment class)
            explored = True

        self.increment_time_step()                    # increment time step as well as update the decay rate
        next_state, reward, done = self.env.step(action)                     # finally, take the action and record the reward
        return current_state, action, reward, next_state, done, explored  # return an experience Tuple


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
