from itertools import product
from math import cos, pi, sqrt
import random

import numpy as np


class FourierFunction(object):
    """This class is responsible for the following.

    1. Use Fourier transformation to turn observation action pairs to feature
       vectors. See
       https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view,
       Page 211 for more information on Fourier transforms.
    2. Map feature vectors to Q values using a linear model.
    3. Update the model using the experience supplied by the agent.
    4. Implement an epsilon greedy policy based on the model.

    This implementation assumes a continuous observation space and a discrete
    action space.
    """

    def __init__(self, env, max_fourier_basis_order,
                 eligibility_traces = None, weights = None,
                 eligibility_traces_file_path = None,
                 weights_file_path = None
                 ):
        """
        Arguments:
        env -- A Gym environment
        max_fourier_basis_order -- The fourier expansion is restricted to this
                                   order
        eligibility_traces -- Eligibility traces for features.
                              If supplied, it should be used as a starting
                              point for further training. If None, then
                              it is initialized to zero.
        weights -- Weights of the linear model. If supplied, these weights would
                   be used to compute the epsilon greedy policy. If None, the
                   weights are initialized to zero.
        eligibility_traces_file_path -- We store eligibility traces in this
                                        file when the save() method is called.
        weights_file_path -- We store weights in this file path when the save()
                             method is called.
        """
        self.env = env
        self.max_fourier_basis_order = max_fourier_basis_order

        ## Compute number of features in this setting
        self.number_of_orders = self.max_fourier_basis_order + 1
        self.observation_space_dimension = (
            self.env.observation_space.low.shape[0]
            )
        self.number_of_features_for_each_action = (
            self.number_of_orders**self.observation_space_dimension
            )
        self.number_of_features = (
            self.env.action_space.n*self.number_of_features_for_each_action
            )

        self.observation_lower_bounds = self.env.observation_space.low
        self.observation_upper_bounds = self.env.observation_space.high

        # If weights is already supplied, use it. Useful for testing saved
        # models.
        if weights is not None:
            self.weights = weights
        # Otherwise, initialize weights to zero.
        else:
            self.weights = np.array([0 for i in range(self.number_of_features)])

        # If eligibility traces is already supplied, use it. Useful for further
        # training from a saved model.
        if eligibility_traces is not None:
            self.eligibility_traces = eligibility_traces
        # Otherwise, initialize eligibility traces to zero
        else:
            self.eligibility_traces = np.array(
                [0 for i in range(self.number_of_features)]
                )

        self.weights_file_path = weights_file_path
        self.eligibility_traces_file_path = (
            eligibility_traces_file_path
            )

        ## Compute learning rate modifiers. According to Konidaris et. al.
        ## (2011), using a learning rate dependent on the basis function
        ## gives better results

        self.learning_rate_modifiers = np.array(
            [0 for i in range(self.number_of_features)]
            )

        # Compute all possible combination of bases
        bases_iterable = product(
            [i for i in range(self.number_of_orders)],
            repeat = self.observation_space_dimension
            )

        for bases in bases_iterable:
            ## Get the index of the feature vector corresponding to this
            ## combination of bases for all actions
            for action in range(self.env.action_space.n):
                learning_rate_modifier_index = (
                    action*self.number_of_features_for_each_action
                    )
                for i in range(len(bases)):
                    learning_rate_modifier_index += (
                        bases[i]*self.number_of_orders**(
                            self.observation_space_dimension - i -1
                            )
                        )

                ## Compute the learning rate modifier
                square_root_sum = sqrt(
                    sum([i**2 for i in bases])
                    )

                self.learning_rate_modifiers[learning_rate_modifier_index] = 1

                if square_root_sum != 0:
                    self.learning_rate_modifiers[
                        learning_rate_modifier_index
                        ]  = (
                        self.learning_rate_modifiers[
                            learning_rate_modifier_index
                            ] /
                        float(square_root_sum)
                        )

    def get_feature_vector_from_observation_and_action(self, observation,
                                                       action
                                                       ):
        """Get a feature vector from a given observation action pair

        Arguments:
        observation -- Gym observation
        action -- Gym action
        """

        feature_vector = np.array([0 for i in range(self.number_of_features)])

        # Compute all possible combination of bases
        bases_iterable = product(
            [i for i in range(self.number_of_orders)],
            repeat = self.observation_space_dimension
            )

        for bases in bases_iterable:

            ## Get the index of the feature vector corresponding to this
            ## combination of bases and this action
            feature_vector_index = (
                action*self.number_of_features_for_each_action
                )
            for i in range(len(bases)):
                feature_vector_index += (
                    bases[i]*self.number_of_orders**(
                        self.observation_space_dimension - i -1
                        )
                    )

            ## Compute the basis function for this combination of bases
            cos_argument = 0

            for i in range(len(bases)):
                cos_argument += (
                    (observation[i] - self.observation_lower_bounds[i]) *
                    bases[i] /
                    float(
                        self.observation_upper_bounds[i] -
                        self.observation_lower_bounds[i]
                        )
                    )

            feature_vector[feature_vector_index] = cos(pi*cos_argument)

        return feature_vector

    def get_q_value_from_observation_and_action(self, observation, action):
        """Return the Q value for an observation action pair

        Arguments:
        observation -- Gym observation
        action -- Gym action
        """

        feature_vector = (
            self.get_feature_vector_from_observation_and_action(
                observation, action
                )
            )

        # Use linear model
        q_value = np.dot(self.weights, feature_vector)

        return q_value

    def update_model(self, discount_factor, lambda_value, learning_rate,
                     observation, action, reward, done, next_observation,
                     next_action
                     ):
        """Update the weights of the model given the agent's experience using
        SARSA(lambda)

        Arguments:
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        lambda_value -- lambda_value -- The Lambda in TD(lambda)
        learning_rate -- Often referred to as alpha in the literature.
        observation -- Last observation seen by agent
        action -- Last action taken by agent
        reward -- Last reward received by agent
        done -- Whether epsiode ended after last action
        next_observation -- The observation after taking the last action
        next_action -- The next action to be taken by the agent according to
                       the epsilon greedy policy
        """

        ## Update eligibility traces

        feature_vector = (
            self.get_feature_vector_from_observation_and_action(
                observation, action
                )
            )

        self.eligibility_traces = (
            discount_factor*lambda_value*self.eligibility_traces +
            feature_vector
            )

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

        ## Update weights)

        self.weights = (
            self.weights +
            learning_rate*td_error *
            np.multiply(self.eligibility_traces, self.learning_rate_modifiers)
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
        """Save the model weights and eligibility traces to file"""

        # If the class wasn't instantiated with file paths, complain.
        if self.weights_file_path is None:
            raise Exception(
                "Weights file path is not specified"
                )

        if self.eligibility_traces_file_path is None:
            raise Exception(
                "Eligibility traces file path is not specified"
                )

        with open(self.weights_file_path, "w") as weights_fh:
            for i in self.weights:
                weights_fh.write("{0}\n".format(i))

        with open(self.eligibility_traces_file_path, "w") as \
                eligibility_traces_fh:
            for i in self.eligibility_traces:
                eligibility_traces_fh.write("{0}\n".format(i))
