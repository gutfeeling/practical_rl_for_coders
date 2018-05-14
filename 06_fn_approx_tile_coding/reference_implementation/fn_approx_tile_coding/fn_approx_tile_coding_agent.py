import datetime
import random

from gym.wrappers import Monitor


class FnApproxAgent(object):
    """Agent using a linear model for mapping observation action pairs to Q
    values.
    """

    def get_epsilon(self, start_epsilon, end_epsilon, observation_number,
                    total_observations
                    ):
        """Get epsilon (probability of random action) given the current observation
        number and the total observations for training

        Arguments:
        start_epsilon -- Training starts with this value of epsilon
        end_epsilon -- Training ends with this value of epsilon
        observation_number -- Current observation number
        total_observations -- Total observations to train on

        Note:
        Epsilon is linearly annealed from start_epsilon to end_epsilon during
        training.
        """

        training_progress_fraction = (
            observation_number / float(total_observations)
            )

        return (
            start_epsilon -
            (start_epsilon - end_epsilon)*training_progress_fraction
            )

    def get_learning_rate(self, start_learning_rate, end_learning_rate,
                          observation_number, total_observations
                          ):
        """Get learning rate given the current observation
        number and the total observations for training

        Arguments:
        start_learning_rate -- Training starts with this value of learning rate
        end_learning_rate -- Training ends with this value of learning rate
        observation_number -- Current observation number
        total_observations -- Total observations to train on

        Note:
        Learning rate is linearly annealed from start_learning_rate to
        end_learning_rate during training.
        """

        training_progress_fraction = (
            observation_number / float(total_observations)
            )

        return (
            start_learning_rate -
            (start_learning_rate - end_learning_rate)*training_progress_fraction
            )

    def test(self, testing_env, total_number_of_episodes, function,
             epsilon, render):

        """Test the agent for a number of episodes and return the
        average reward per episode

        testing_env -- A Gym environment (wrapped or vanilla) used for testing
        total_number_of_episodes -- Test for this many episodes
        function -- An instance of the class implemnenting the
                    function approximation model
        render -- A Boolean indicating whether to render the environment or not
        """

        # Total rewards obtained over all testing episodes
        total_rewards = 0

        for episode_number in range(total_number_of_episodes):

            # Start an episode
            observation = testing_env.reset()

            while True:

                # If render is True, show the current situation
                if render:
                    testing_env.render()

                action = function.get_action(observation, epsilon)

                observation, reward, done, info = (
                    testing_env.step(action)
                    )

                total_rewards += reward

                if done:
                    break

        testing_env.close()

        # Compute average reward per episode
        average_reward = total_rewards/float(total_number_of_episodes)
        return average_reward

    def train(self, function, discount_factor, start_learning_rate,
              end_learning_rate, start_epsilon,
              end_epsilon, learning_env, testing_env,
              total_observations, test_interval,
              total_number_of_testing_episodes,
              gym_training_logs_directory_path, gym_testing_logs_directory_path,
              weight_saving_interval
              ):
        """Train the agent

        function -- An instance of the class implemnenting the
                    function approximation model
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        start_learning_rate -- Learning rate at the start of training
        end_learning_rate -- Learning rate at the end of training
        start_epsilon -- Probability of random actions at start of training
        end_epsilon -- Probability of random actions at end of training
        learning_env -- A Gym environment (wrapped or vanilla) used for learning
        testing_env -- A Gym environment (wrapped or vanilla) used for testing.
        total_observations -- Train till this observation number
        test_interval -- Test after this many observations
        total_number_of_testing_episodes -- Number of episodes to test the agent
                                            in every testing round
        gym_training_logs_directory_path - Directory to save automatic Gym logs
                                           related to training. We save the
                                           rewards for every learning episode.
        gym_testing_logs_directory_path - Directory to save automatic Gym logs
                                          related to testing. We save a video
                                          for the first test episode.
        weight_saving_interval -- Save the model weights (i.e. write to file)
                                  after this many observations.
        """

        # This keeps track of the number of observations made so far
        observation_number = 0

        # Keep count of the episode number
        episode_number = 1

        # The learning env should always be wrapped by the Monitor provided
        # by Gym. This lets us automatically save the rewards for every episode.

        learning_env = Monitor(
            learning_env, gym_training_logs_directory_path,
            # Don't want video recording during training, only during testing
            video_callable = False,
            # Write after every reset so that we don't lose data for
            # prematurely interrupted training runs
            write_upon_reset = True,
            )

        while observation_number < total_observations:

            # initialize environment
            observation = learning_env.reset()

            total_rewards_obtained_in_this_episode = 0

            # get the epsilon-greedy action for the first observation
            epsilon = self.get_epsilon(
                start_epsilon, end_epsilon, observation_number,
                total_observations
                )

            action = function.get_action(observation, epsilon)

            # Execute an episode
            while True:

                # take the action determined by the epsilon-greedy policy
                next_observation, reward, done, info = learning_env.step(action)

                # Determine the next action. This is required for the
                # model update.
                epsilon = self.get_epsilon(
                    start_epsilon, end_epsilon, observation_number,
                    total_observations
                    )

                next_action = function.get_action(
                    next_observation, epsilon
                    )

                # Learning rate is linearly annealed during training. Determine
                # the current value.
                learning_rate = self.get_learning_rate(
                    start_learning_rate, end_learning_rate, observation_number,
                    total_observations
                    )

                # Update the model
                function.update_weights(
                    discount_factor, learning_rate, observation, action,
                    reward, done, next_observation, next_action,
                    )

                observation = next_observation
                action = next_action

                observation_number += 1
                # Test the current performance after every test_interval
                if observation_number % test_interval == 0:
                    # The testing env is also wrapped by a Monitor so that we
                    # can take automatic videos during testing. We will take a
                    # video for the very first testing episode.

                    video_callable = lambda count : count == 0

                    # Since the environment is closed after every testing round,
                    # the video for different testing round will end up having
                    # the same name! To differentiate the videos, we pass
                    # an unique uid parameter.

                    monitored_testing_env = Monitor(
                        testing_env, gym_testing_logs_directory_path,
                        video_callable = video_callable,
                        resume = True,
                        uid = observation_number / test_interval
                        )

                    # Run the test
                    average_reward = self.test(
                        monitored_testing_env,
                        total_number_of_episodes =
                            total_number_of_testing_episodes,
                        function = function,
                        epsilon = 0,
                        render = False
                        )
                    print(
                        "[{0}] Episode number : {1}, Observation number : {2} "
                        "Average reward (100 eps) : {3}".format(
                            datetime.datetime.now(), episode_number,
                            observation_number, average_reward
                            )
                        )

                total_rewards_obtained_in_this_episode += reward

                if done:

                    episode_number +=1

                    # Save table to file at regular intervals
                    if episode_number % weight_saving_interval == 0:
                        function.save()

                    break

            print(
                "[{0}] Episode number : {1}, Obervation number: {2}, "
                "Reward in this episode : {3}, Epsilon : {4}, "
                "Learning rate : {5}".format(
                    datetime.datetime.now(), episode_number - 1,
                    observation_number, total_rewards_obtained_in_this_episode,
                    epsilon, learning_rate
                    )
                )

        learning_env.close()

        # There's a bug in the Gym Monitor. The Monitor's close method does not
        # close the wrapped environment. This makes the script exit with an
        # error if the environment is being rendered at some point. To make
        # this error go away, we have to close the unwrapped testing
        # environment. The learning environment is not being rendered, so we
        # don't need to bother about that.
        testing_env.env.close()
