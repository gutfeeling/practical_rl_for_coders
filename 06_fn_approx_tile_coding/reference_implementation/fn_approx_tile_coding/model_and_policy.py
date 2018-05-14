import random

from tile_coding.tiles import tiles


class TileCodingLinearFunction(object):
    """This class is responsible for the following.

    1. Use Tile Coding to turn observation action pairs to feature vectors. See
       https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view,
       Page 217 for more information on Tile Coding.
    2. Map feature vectors to Q values using a linear model.
    3. Update the model using the experience supplied by the agent.
    4. Implement an epsilon greedy policy based on the model.

    This implementation assumes a continuous observation space and a discrete
    action space.
    """

    def __init__(self, env, number_of_tiles, number_of_tilings, weights = None,
                 weights_file_path = None):
        """
        Arguments:
        env -- A Gym environment
        number_of_tiles -- Number of tiles to use for each observation dimension
        number_of_tilings -- Number of overlapping tilings to use
        weights -- Weights of the linear model. If supplied, these weights would
                   be used to compute the epsilon greedy policy. If None, the
                   weights are initialized to zero.
        weights_file_path -- We store weights in this file path when the save()
                             method is called.
        """
        self.env = env
        self.number_of_tiles = number_of_tiles
        self.number_of_tilings = number_of_tilings

        # Memory size is essentially the total number of features. Increase
        # this to prevent collisions i.e. when distinct tiles maps to the
        # same feature.
        self.memory_size = 10*self.number_of_tilings*self.number_of_tiles**2

        self.observation_lower_bounds = self.env.observation_space.low
        self.observation_upper_bounds = self.env.observation_space.high

        # If weights is already supplied, use it. Useful for testing saved
        # models.
        if weights is not None:
            self.weights = weights
        # Otherwise, initialize weights to zero.
        else:
            self.weights = [0 for i in range(self.memory_size)]

        self.weights_file_path = weights_file_path

    def get_feature_vector_from_observation_and_action(self, observation,
                                                       action
                                                       ):
        """Get a feature vector from a given observation action pair

        Arguments:
        observation -- Gym observation
        action -- Gym action
        """

        # The observations are scaled to fall between 0 and the number of tiles
        scaled_observation = [
            self.number_of_tiles*observation[i] /
            (
                self.observation_upper_bounds[i] -
                self.observation_lower_bounds[i]
                )
            for i in range(len(observation))
            ]

        # Use the tiles function from Sutton to get a sparse feature vector.
        # If the vector is [234, 678, 123, 678, 890, 245, 800], it means
        # that feature number 234, 678, 123, 678, 890, 245 and 800 are all 1.
        # The rest of the features are zero.
        sparse_feature_vector = tiles(
            numtilings = self.number_of_tilings,
            memctable = self.memory_size,
            floats = scaled_observation,
            ints = [action,]
            )

        return sparse_feature_vector

    def get_q_value_from_observation_and_action(self, observation, action):
        """Return the Q value for an observation action pair

        Arguments:
        observation -- Gym observation
        action -- Gym action
        """

        sparse_feature_vector = (
            self.get_feature_vector_from_observation_and_action(
                observation, action
                )
            )

        # Use linear model
        q_value = sum([self.weights[i] for i in sparse_feature_vector])

        return q_value

    def update_weights(self, discount_factor, learning_rate, observation,
                       action, reward, done, next_observation, next_action
                       ):
        """Update the weights of the model given the agent's experience using
        SARSA(0)

        Arguments:
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        learning_rate -- Often referred to as alpha in the literature.
        observation -- Last observation seen by agent
        action -- Last action taken by agent
        reward -- Last reward received by agent
        done -- Whether epsiode ended after last action
        next_observation -- The observation after taking the last action
        next_action -- The next action to be taken by the agent according to
                       the epsilon greedy policy
        """

        ## Compute TD error

        q_value = self.get_q_value_from_observation_and_action(
            observation, action
            )

        next_q_value = 0

        if not done:
            next_q_value = self.get_q_value_from_observation_and_action(
                next_observation, next_action
                )

        td_error = reward + discount_factor*next_q_value - q_value

        ## Update weights using TD error

        sparse_feature_vector = (
            self.get_feature_vector_from_observation_and_action(
                observation, action
                )
            )

        # Note that we need to divide the vanilla learning rate by the number of
        # tilings to get the actual learning rate.
        for i in sparse_feature_vector:
            self.weights[i] += (
                learning_rate*td_error/float(self.number_of_tilings)
                )

    def get_action(self, observation, epsilon):
        """Return the epsilon greedy action

        Arguments:
        observation -- Gym observation
        epsilon -- Probability for taking a random action
        """

        # With probability epsilon, take a random action
        if random.random() < epsilon:
            return self.env.action_space.sample()

        # Else, be greedy
        q_values_for_this_observation = {}

        for action in range(self.env.action_space.n):
            q_values_for_this_observation[action] = (
                self.get_q_value_from_observation_and_action(
                    observation, action
                    )
                )

        # Greedy means taking the action with maximum Q
        max_q_value = max(q_values_for_this_observation.values())

        actions_with_max_q_value = [
            action
            for action, q_value in q_values_for_this_observation.items()
            if q_value == max_q_value
            ]

        # If there is a tie between two actions
        # (i.e. they both have the same Q value),
        # choose between them randomly
        return random.choice(actions_with_max_q_value)

    def save(self):
        """Save the model weights to file"""

        # If the class wasn't instantiated with file paths, complain.
        if self.weights_file_path is None:
            raise Exception(
                "Weights file path is not specified"
                )

        with open(self.weights_file_path, "w") as weights_fh:
            for i in self.weights:
                weights_fh.write("{0}\n".format(i))
