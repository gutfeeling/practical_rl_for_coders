import datetime
import random

from gym.wrappers import Monitor

from glie_monte_carlo.models import QValueNotFoundError

class GLIEMonteCarloAgent(object):
    """GLIE Monte Carlo agent for environments with one dimensional discrete action
    space
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
        training. This is why this algorithm is called GLIE (Greedy in the Limit
        of Infinite Exploration).
        """

        training_progress_fraction = (
            observation_number / float(total_observations)
            )

        return (
            start_epsilon -
            (start_epsilon - end_epsilon)*training_progress_fraction
            )

    def get_action(self, env, table, observation, epsilon):
        """Get the action for an epsilon greedy policy given an observation

        Arguments:
        env -- The Gym environment
        table -- Table storing Q values for state action pairs
        observation -- Gym observation
        epsilon -- Probability for taking a random action
        """

        # With probability epsilon, take a random action
        if random.random() < epsilon:
            return env.action_space.sample()

        # Else, be greedy
        q_values_for_this_observation = {}

        for action in range(env.action_space.n):
            try:
                q_value = table.get_q_value(observation, action)
                q_values_for_this_observation[action] = q_value
            except QValueNotFoundError:
                # Assume all unknown Q values to be 0
                q_values_for_this_observation[action] = 0

        # Greedy means taking the action with maximum Q
        max_q_value = max(q_values_for_this_observation.values())

        actions_with_max_q_value = [
            action
            for action, q_value in q_values_for_this_observation.items()
            if q_value == max_q_value
            ]

        # If there is a tie between two actions (i.e. they both have the same Q value),
        # choose between them randomly
        return random.choice(actions_with_max_q_value)

    def update_table(self, table, discount_factor, episode_history):
        """Update the table storing Q values and visit numbers for state action
        pairs at the end of an episode

        Arguments:
        table -- Table storing Q values and visit numbers for state action
                 pairs
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        episode_history -- History of an episode. Has the form
                           [
                                {
                                    "observation" : observation,
                                    "action" : action,
                                    "reward" : reward
                                    }, ...
                                ]
        """

        sum_of_discounted_rewards = 0

        for event in reversed(episode_history):

            observation = event["observation"]
            action = event["action"]
            reward = event["reward"]

            # calculate sum of discounted reward starting recursively

            sum_of_discounted_rewards = (
                reward + discount_factor*sum_of_discounted_rewards
                )

            # update number of visits to this state and Q value for the state

            visit_number = table.get_visit_number(
                observation, action
                )
            visit_number += 1
            table.update_visit_number(observation, action, visit_number)

            try:
                q_value = table.get_q_value(observation, action)
                updated_q_value = q_value + (
                    sum_of_discounted_rewards - q_value
                    ) / float(visit_number)
            except QValueNotFoundError:
                updated_q_value = sum_of_discounted_rewards

            table.update_q_value(observation, action, updated_q_value)


    def test(self, testing_env, total_number_of_episodes, table,
             epsilon, render):
        """Test the GLIE Monte Carlo agent for a number of episodes and return the
        average reward per episode

        testing_env -- A Gym environment (wrapped or vanilla) used for testing
        total_number_of_episodes -- Test for this many episodes
        table -- Table storing learned Q values for state action pairs
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

                action = self.get_action(
                    testing_env, table, observation, epsilon
                    )

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

    def train(self, table, discount_factor, start_epsilon,
              end_epsilon, learning_env, testing_env,
              total_observations, test_interval,
              total_number_of_testing_episodes,
              gym_training_logs_directory_path, gym_testing_logs_directory_path,
              table_saving_interval
              ):
        """Train the GLIE Monte Carlo agent

        table -- Table storing Q values and visit numbers for state action pairs
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
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
        table_saving_interval -- Save the table (i.e. write the table to file) after
                                 this many observations.
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

            # Need a list to hold the relevant information for this episode.
            # This will be used in updating Q values.
            # structure : [
            #   {
            #       "observation" : observation,
            #       "action" : action,
            #       "reward" : immediate reward
            #       },...
            #    ]
            episode_history = []

            # initialize environment
            observation = learning_env.reset()

            total_rewards_obtained_in_this_episode = 0

            # Execute an episode
            while True:

                epsilon = self.get_epsilon(
                    start_epsilon, end_epsilon, observation_number,
                    total_observations
                    )

                # use the epsilon-greedy policy to choose an action
                action = self.get_action(
                    learning_env, table, observation, epsilon
                    )

                # take the action determined by the epsilon-greedy policy
                next_observation, reward, done, info = learning_env.step(action)

                # add the current state and resulting reward to history
                episode_history.append(
                    {
                        "observation" : observation,
                        "action" : action,
                        "reward" : reward
                        }
                    )

                observation = next_observation

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
                        table = table,
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
                    # episode has ended, update Q and N values
                    self.update_table(
                        table, discount_factor, episode_history
                        )
                    episode_number += 1
                    # save the table at regular intervals
                    if episode_number % table_saving_interval == 0:
                        table.save()
                    break

            print(
                "[{0}] Episode number : {1}, Obervation number: {2}, "
                "Reward in this episode : {3}, Epsilon : {4}".format(
                    datetime.datetime.now(), episode_number - 1,
                    observation_number, total_rewards_obtained_in_this_episode,
                    epsilon
                    )
                )

        learning_env.close()

        # There's a bug in the Gym Monitor. The Monitor's close method does not
        # close the wrapped environment. This makes the script exit with an
        # error if the environment is being rendered at some point. To make
        # this error go away, we have to close the unwrapped testing
        # environment. The learning environment is not being rendered, so we
        # do the same for that.
        testing_env.env.close()
