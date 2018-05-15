import random

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


class NeuralNetwork(object):
    """This class is responsible for the following.

    1. Map observations to Q values using a neural network.
    3. Update the model using the experience supplied by the agent.
    4. Implement an epsilon greedy policy based on the model.

    This implementation assumes a continuous observation space and a discrete
    action space.
    """

    def __init__(self, env, lr, model = None, model_saving_file_path = None,
                 model_saving_interval = 1, training_logs_file_path = None
                 ):
        """
        Arguments:
        env -- A Gym environment (can be vanilla or wrapped). We need this
               for infering input and output dimensions of the model.
        lr -- Learning rate
        model -- A compiled Keras model. If None, then the model is created.
        training_logs_file_path -- The file where we should save the logs for
                                   training. This is useful for monitoring loss
                                   etc.
        model_saving_file_path -- The model will be saved to this filepath after
                                  a certain number of epochs of training
        model_saving_interval -- The model will be saved to
                                 model_saving_file_path after this many epochs.
        """
        self.env = env
        self.lr = lr
        self.model_saving_file_path = model_saving_file_path
        self.model_saving_interval = model_saving_interval
        self.training_logs_file_path = training_logs_file_path

        # If model is already supplied, use it. Useful for testing saved
        # models.
        if model is not None:
            self.model = model
        # Otherwise, initialize weights to zero.
        else:
            # If no model is specified, create a new model
            self.model = Sequential()
            self.model.add(
                Dense(
                    16,
                    input_dim = self.env.observation_space.shape[0],
                    activation = "relu"
                    )
                )
            self.model.add(
                Dense(16, activation = "relu")
                )
            self.model.add(
                Dense(self.env.action_space.n, activation = "linear")
                )
            self.model.compile(loss = "mse", optimizer = Adam(self.lr))

        # Define the callbacks for training. The callbacks write logs for the
        # training steps.

        self.callbacks = []
        # Add the callback for saving model data after regular intervals
        if self.model_saving_file_path is not None:
            callback = ModelCheckpoint(
                self.model_saving_file_path,
                period = self.model_saving_interval
                )
            self.callbacks.append(callback)

        # Add the callback for saving logs during training
        if self.training_logs_file_path is not None:
            callback = CSVLogger(self.training_logs_file_path, append = True)
            self.callbacks.append(callback)

    def update_model(self, discount_factor, observation,
                     action, reward, done, next_observation, next_action
                     ):
        """Update the weights of the model given the agent's experience using
        SARSA(0)

        Arguments:
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        observation -- Last observation seen by agent
        action -- Last action taken by agent
        reward -- Last reward received by agent
        done -- Whether epsiode ended after last action
        next_observation -- The observation after taking the last action
        next_action -- The next action to be taken by the agent according to
                       the epsilon greedy policy
        """

        ## Get predicted Q values

        model_input = np.array([observation])
        predictions = self.model.predict(model_input)

        ## Get target Q value according to SARSA(0)

        next_q_value = 0

        if not done:
            next_model_input = np.array([next_observation])
            next_predictions = self.model.predict(next_model_input)
            next_q_value = next_predictions[0][next_action]

        target_q_value = reward + discount_factor*next_q_value

        ## Compute targets

        # To train the neural network, we use a target that differs from
        # the prediction only for the action that was actually taken. For
        # the other actions, we use the same value as the prediction. Since
        # we are using MSE loss, the loss from the other actions are then zero.

        targets = predictions
        targets[0][action] = target_q_value

        ## Update model

        self.model.fit(
            model_input,
            targets,
            epochs = 1,
            callbacks = self.callbacks,
            verbose = False,
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
        model_input = np.array([observation])
        predictions = self.model.predict(model_input)

        q_values_for_this_observation = {
            i : predictions[0][i] for i in range(predictions.shape[1])
            }

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
