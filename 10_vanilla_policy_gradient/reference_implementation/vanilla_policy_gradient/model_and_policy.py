from math import exp, sqrt
import random

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense
from keras.models import Input, Model, Sequential
from keras.optimizers import Adam
import numpy as np


class ActorCritic(object):
    """This class implements the default actor critic architecture for modeling
    the environment and choosing a policy.

    You can always write your own implementation of the actor critic
    architecture that differs from the default implementation. Just override the
    get_action(), update_mode() and save() methods of this class, while keeping
    the methods signatures same. Then import your new class and use it in
    run_vanilla_policy_gradient_agent.py instead of this default actor critic
    architecture.
    """

    def __init__(self, env, lr_critic,
                 actor_weights = None, critic_model = None,
                 critic_training_logs_file_path = None,
                 actor_weights_saving_file_path = None,
                 critic_model_saving_file_path = None,
                 critic_model_saving_interval = 1,
                 ):
        """
        Arguments:
        env -- A Gym environment (can be vanilla or wrapped). We need this
               for infering input and output dimensions of the model.
        lr_critic -- Learning rate of the critic
        actor_weights -- Weights of the actor. If not None, the supplied
                         weights are used to compute policy and for further
                         training. If None, the actor weights are initialized
                         to zero.
        critic_model -- A compiled Keras model for the critic. If None, then
                        the critic model is created. If not None, then the
                        supplied model is used for further training.
        critic_training_logs_file_path -- The file where we should save the Keras
                                          logs for critic training. This is
                                          useful for monitoring loss etc.
        actor_weights_saving_file_path -- The actor weights will be saved to this
                                          filepath when the save() method is
                                          called
        critic_model_saving_file_path -- The critic model will be saved to this
                                         filepath after a certain number of
                                         epochs of training
        critic_model_saving_interval -- The critic model will be saved to
                                        critic_model_saving_file_path
                                        after this many epochs

        Notes about the model:
        We will use a MLP (Multi Layer Perceptron) with two hidden layers
        (with 64 units each) as the critic model in our reference implementation.
        We will use ReLU activation for the hidden layer.

        The model assumes that we are dealing with continuous environments
        like LunarLander-v2, where the observation shape is (n,),
        where n can be any any integer. It also assumes a discrete
        action space.

        The actor uses a softmax policy to output probabilities
        for each action.
        """

        self.env = env
        self.lr_critic = lr_critic
        self.critic_training_logs_file_path = critic_training_logs_file_path
        self.actor_weights_saving_file_path = actor_weights_saving_file_path
        self.critic_model_saving_file_path = critic_model_saving_file_path
        self.critic_model_saving_interval = critic_model_saving_interval

        # If actor weights have been provided, use it. Useful for testing
        # saved models.
        if actor_weights is not None:
            self.actor_weights = actor_weights
        # Else, initialize actor weights to zero.
        else:
            # The weights should have the same shape as the feature vector.
            # In this implementation, we use a feature vector that has the
            # following form:
            # concatenate_over_all-actions(f(a_i)*observation),
            # where f(a_i) is 1 only when a_i is the current action, 0
            # otherwise. This vector has length
            # number_of_actions*observation_length, so the weights should
            # have this length too.
            self.actor_weights = np.array(
                [
                    0. for i in range(
                        self.env.observation_space.shape[0] *
                        self.env.action_space.n
                        )
                    ]
                )

        # If model is supplied, use it for further training.
        if critic_model is not None:
            self.critic_model = critic_model
        # Else, create a new model.
        else:
            self.critic_model = Sequential()
            self.critic_model.add(
                Dense(
                    64,
                    input_dim = self.env.observation_space.shape[0],
                    activation = "relu"
                    )
                )
            self.critic_model.add(
                Dense(64, activation = "relu")
                )
            self.critic_model.add(
                Dense(1, activation = "linear")
                )
            self.critic_model.compile(
                loss = "mse", optimizer = Adam(lr = self.lr_critic)
                )

        # Define the callbacks for training. The callbacks write logs for the
        # critic training steps.

        self.callbacks = []
        # Add the callback for saving critic model data after regular intervals
        if self.critic_model_saving_file_path is not None:
            callback = ModelCheckpoint(
                self.critic_model_saving_file_path,
                period = self.critic_model_saving_interval
                )
            self.callbacks.append(callback)

        # Add the callback for saving logs during critic training
        if self.critic_training_logs_file_path is not None:
            callback = CSVLogger(
                self.critic_training_logs_file_path, append = True
                )
            self.callbacks.append(callback)

    def get_feature_vector_from_observation_and_action(self, observation,
                                                       action
                                                       ):
        """Get a feature vector from a given observation action pair

        Arguments:
        observation -- Gym observation
        action -- Gym action

        Notes:
        In this implementation, we use a feature vector that has the
        following form:
        concatenate_over_all-actions(f(a_i)*observation),
        where f(a_i) is 1 only when a_i is the current action, 0 otherwise.
        """

        # Initialize to zeros
        feature_vector = np.array(
            [
                0. for i in range(
                    self.env.observation_space.shape[0] *
                    self.env.action_space.n
                    )
                ]
            )

        # Replace the indices corresponding to action with the actual
        # observation. Leave everything else as zero.
        feature_vector[
            self.env.observation_space.shape[0]*action :
            self.env.observation_space.shape[0]*(action + 1)
            ] = observation

        return feature_vector

    def get_normalized_softmax_probabilities_from_observation(self,
                                                              observation
                                                              ):
        """Get the softmax policy (probabilities for takingeach action) from the
        observation

        Arguments:
        observation -- Gym observation
        """

        probabilities = {}

        for action in range(self.env.action_space.n):
            feature_vector = (
                self.get_feature_vector_from_observation_and_action(
                    observation, action
                    )
                )

            # Non-normalized Softmax probability
            probabilities[action] = exp(
                    np.dot(self.actor_weights, feature_vector)
                    )


        ## Normalize the probabilities

        normalization_factor = sum(probabilities.values())

        for action in probabilities:
            probabilities[action] /= normalization_factor

        return probabilities

    def get_action(self, observation):
        """Return the softmax action, given an observation

        Arguments:
        observation -- Gym observation
        """

        probabilities = (
            self.get_normalized_softmax_probabilities_from_observation(
                observation
                )
            )

        actions = []
        probabilities_list = []

        for action in probabilities:
            actions.append(action)
            probabilities_list.append(probabilities[action])

        # Sample from the actions using the softmax probabilities. Since
        # this is inherently probabilisitic, the agent will always explore.
        action = np.random.choice(actions, p = probabilities_list)

        return action

    def update_model(self, discount_rate, actor_learning_rate, observation,
                     action, reward, done, next_observation, next_action
                     ):
        """Update the weights of the model given the agent's experience using
        SARSA(0)

        Arguments:
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        actor_learning_rate -- Learning rate of Actor.
        observation -- Last observation seen by agent
        action -- Last action taken by agent
        reward -- Last reward received by agent
        done -- Whether epsiode ended after last action
        next_observation -- The observation after taking the last action
        next_action -- The next action to be taken by the agent according to
                       the epsilon greedy policy
        """

        ## Compute TD error

        critic_model_input = np.array([observation])
        prediction = self.critic_model.predict(critic_model_input)
        value = prediction[0][0]

        next_value = 0
        if not done:
            next_critic_model_input = np.array([next_observation])
            next_prediction = self.critic_model.predict(next_critic_model_input)
            next_value = next_prediction[0][0]

        target_value = reward + discount_rate*next_value
        td_error = target_value - value

        targets = np.array([[target_value,]])

        ## Update critic

        self.critic_model.fit(
            critic_model_input,
            targets,
            epochs = 1,
            callbacks = self.callbacks,
            verbose = False,
            )

        ## Update actor model using the Vanilla Policy Gradient update

        probabilities = (
            self.get_normalized_softmax_probabilities_from_observation(
                observation
                )
            )

        feature_vector_for_actions = []
        for i in range(self.env.action_space.n):
            feature_vector_for_this_action = (
                    self.get_feature_vector_from_observation_and_action(
                        observation, i
                        )
                    )
            feature_vector_for_actions.append(feature_vector_for_this_action)


        for i in range(self.actor_weights.shape[0]):
            score = feature_vector_for_actions[action][i]
            for j in range(self.env.action_space.n):
                score -= probabilities[j]*feature_vector_for_actions[j][i]

            self.actor_weights[i] += actor_learning_rate*td_error*score

    def save(self):
        """Save the actor weights to file"""

        # If the class wasn't instantiated with file paths, complain.
        if self.actor_weights_saving_file_path is None:
            raise Exception(
                "Weights file path is not specified"
                )

        with open(self.actor_weights_saving_file_path, "w") \
                as actor_weights_saving_fh:
            for i in self.actor_weights:
                actor_weights_saving_fh.write("{0}\n".format(i))
