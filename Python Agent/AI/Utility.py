import numpy as np

from AI.Agent import Agent
from AI.EnvironmentWrapper import EnvironmentWrapper
from AI.Memory import Memory
from AI.NetworkTracker import NetworkTracker


class Utility:
    def __init__(self):
        pass

    def train_agent(render=False, verbose=False, num_episodes=1500,
                    discount = 0.95, batch_size = 64, N = 40, memory_size = 1024,
                    eps_decay_rate=0.9975, max_exp_rate=1.0, min_exp_rate=0.05):
        # get all the hyperparameters in one place!

        # stores all the total reward per episode
        training_stats = []

        # initialise your environment
        env = EnvironmentWrapper()

        # initialise your policy and target networks
        net = NetworkTracker(env)

        # initialise your agent that will follow the epsilon greedy strategy
        agent = Agent(env, net, eps_decay_rate=eps_decay_rate, max_exp_rate=max_exp_rate,min_exp_rate=min_exp_rate )

        # initialise experience replay memory
        memory = Memory(memory_size)

        for episode_count in range(num_episodes):
            state = env.reset()

            # uncomment if you want to start the environmet with a random move
            # state = env.step(env.get_random_action())[0]
            cumulative_reward = 0
            exploit_times = 0
            while not env.is_done():
                current_state, action, reward, next_state, done = agent.take_action(state)
                Cumulative_reward += reward
                experience = current_state, action, reward, next_state, done, explored
                state = next_state
                memory.push(experience)
                if not explored:
                    exploit_times += 1

                # maybe add a clause which says don't push experience we the whole Cumulative_reward is not more than 20 or any suitable number
            agent.update_epsilon()
            if memory.is_usable(batch_size):

                experience_batch = memory.sample(batch_size)
                states, actions, rewards, next_states, done_tensor = extract_tensors(experience_batch)

                target_batch = get_target_batch(states, actions, rewards, next_states, done_tensor, net, discount)


                net.fit(states, target_batch)

            if (episode_count + 1) % N == 0:
                net.clone_policy()

            training_stats.append(cumulative_reward)


            if verbose:
                print("Episode Count: ", episode_count, "\t Cumulative Reward: ", Cumulative_reward, "\t eps: ", exploit_times/Cumulative_reward )
            if episode_memory.count(500) > 10:
                break


        epochs = list(range(len(training_stats)))

        plt.clf
        plt.plot(epochs, training_stats, 'b', label='f')
        env.close()

        return epochs, training_stats


    def extract_tensors(self, sample):
        states = np.asarray([i[0] for i in sample])
        actions = np.asarray([i[1] for i in sample])
        rewards = np.asarray([i[2] for i in sample])
        next_states = np.asarray([i[3] for i in sample])
        done_tensor = np.asarray([i[4] for i in sample])
        return states, actions, rewards, next_states, done_tensor

    def get_target_batch(self, states, actions, rewards, next_states, dones, net, gamma):
        assert actions.ndim == 1
        assert rewards.ndim == 1
        assert dones.ndim == 1
        assert len(actions) == len(rewards) == len(dones) == len(states) == len(next_states)
        target_q_values = net.get_q_values_for_batch(states)
        targets = rewards + gamma * (np.max(net.get_target_tensor(next_states), axis=1))
        for i in range(len(targets)):
            if dones[i]:
                targets[i] = rewards[i]
        for index in range(len(target_q_values)):
            target_q_values[index][actions[index]] = targets[index]

        return target_q_values
