import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from AI import *
from AI.Agent import Agent
from AI.EnvironmentWrapper import EnvironmentWrapper
from AI.RNNEnvironmentWrapper import RNNEnvironmentWrapper
from AI.Memory import Memory
from AI.NetworkTracker import NetworkTracker
from AI.RnnNetworkTracker import RnnNetworkTracker

def extract_tensors(sample):
    states = np.asarray([i[0] for i in sample])
    actions = np.asarray([i[1] for i in sample])
    rewards = np.asarray([i[2] for i in sample])
    next_states = np.asarray([i[3] for i in sample])
    done_tensor = np.asarray([i[4] for i in sample])
    return states, actions, rewards, next_states, done_tensor


def get_target_batch(states, actions, rewards, next_states, dones, net, gamma):
    assert actions.ndim == 1
    assert rewards.ndim == 1
    assert dones.ndim == 1
    assert len(actions) == len(rewards) == len(dones) == len(states) == len(
        next_states)
    # print("Batch States: {}".format(states))
    # print("Batch States Shape:{}".format(states.shape))
    target_q_values = net.get_q_values_for_batch(
        states)  # get the q values from the current states
    targets = rewards + gamma * (np.max(net.get_target_tensor(next_states),
                                        axis=1))  # get the target values for the state action pairs.
    for i in range(len(targets)):  # change the targets for state which ended an episode
        if dones[i]:
            targets[i] = rewards[i]

    for index in range(len(target_q_values)):
        target_q_values[index][actions[index]] = targets[
            index]  # assign the targets to the corresponding state action pairs

    return target_q_values

def train_agent(contd=True, verbose=False, num_episodes=1500,
                discount=0.95, batch_size=64, N=40, memory_size=1024,
                eps_decay_rate=0.9975, max_exp_rate=1.0, min_exp_rate=0.05):
    # get all the hyperparameters in one place!

    # stores all the total reward per episode
    training_stats = []

    # initialise your environment
    env = RNNEnvironmentWrapper()

    # initialise your policy and target networks
    # change here the model you want to train
    net = RnnNetworkTracker(env, source=contd)
    print(net.get_model_summary())

    # initialise your agent that will follow the epsilon greedy strategy
    agent = Agent(env, net, eps_decay_rate=eps_decay_rate,
                  max_exp_rate=max_exp_rate, min_exp_rate=min_exp_rate)

    # initialise experience replay memory
    memory = Memory(memory_size)

    # graph display init code
    epochs = []
    # % matplotlib notebook
    plt.rcParams['animation.html'] = 'jshtml'
    fig = plt.figure()
    subplot = fig.add_subplot(111)

    for episode_count in range(num_episodes):
        # uncomment if you want to start the environmet with a random move
        # state = env.step(env.get_random_action())[0]

        # keeps track of the total reward that we got for this episode

        valid_episode = False
        # check if the environment is available to run
        while not valid_episode:
            stuck_counter = 0
            cumulative_reward = 0
            counter = 0
            state = env.reset()
            while not env.is_done():  # run the environment for one episode
                counter += 1
                current_state, action, reward, next_state, done = agent.take_action(
                    state)  # let the agent take an action for one time step
                # print("Current State: {}".format(current_state))
                cumulative_reward += reward
                experience = current_state, action, reward, next_state, done  # experience tuple
                state = next_state  # update the current state
                memory.push(experience)  # push the experience in memory

            if counter > 3:
                valid_episode = True

        # update the exploration rate of the agent after each episode
        agent.update_epsilon()

        if memory.is_usable(batch_size):
            experience_batch = memory.sample(batch_size)  # sample randomly from memory
            states, actions, rewards, next_states, done_tensor = extract_tensors(
                experience_batch)  # unzips the tensors
            # print("Experience Batch:{}".format(experience_batch))

            target_batch = get_target_batch(states, actions, rewards,
                                            next_states, done_tensor, net,
                                            discount)  # get a batch of target values to fit against

            net.fit(states, target_batch)

        training_stats.append(cumulative_reward)
        epochs.append(episode_count)

        # save the model
        if (episode_count + 1) % N == 0:
            net.clone_policy()  # clone the target policy every N episodes.

        if (episode_count + 1) % 10 == 0:
            subplot.plot(epochs, training_stats, color='b')
            fig.canvas.draw()

        f = open("TalesRNNstats50.txt", "a")
        f.write("{},{},{}\n".format(episode_count, cumulative_reward, agent.exp_rate))
        f.close()

        if verbose:
            print("Episode Count: ", episode_count, "\t Cumulative Reward: ",
                  cumulative_reward, "\t eps: ", agent.exp_rate)

    return epochs, training_stats, net

if __name__ == '__main__':
    train_agent(contd=False,
                verbose=True,
                num_episodes=10000,
                discount=0.99,
                batch_size=64,
                N=25,  # how often to clone the target policy
                memory_size=5196,
                eps_decay_rate=0.999,
                max_exp_rate=0.05,
                min_exp_rate=0.05)
