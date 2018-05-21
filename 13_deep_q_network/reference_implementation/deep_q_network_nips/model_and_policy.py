import random

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np


class NeuralNetwork(object):
    """This class is responsible for the following.

    1. Get features from observation
    1. Map features to Q values using a neural network.
    3. Update the model using the experience supplied by the agent.
    4. Implement an epsilon greedy policy based on the model.

    This implementation assumes the Atari environments in OpenAI Gym.
    """

    def __init__(self, env, lr, rmsprop_rho, minibatch_size, model = None,
                 model_saving_file_path = None, model_saving_interval = 1,
                 training_logs_file_path = None
                 ):
        """
        Arguments:
        env -- A Gym environment (can be vanilla or wrapped). We need this
               for infering input and output dimensions of the model.
        lr -- Learning rate in RMSprop
        rmsprop_rho -- Rho in RMSProp, see https://keras.io/optimizers/#rmsprop
        model -- A compiled Keras model. If None, then the model is created.
        training_logs_file_path -- The file where we should save the logs for
                                   training. This is useful for monitoring loss
                                   etc.
        model_saving_file_path -- The model will be saved to this filepath after
                                  a certain number of epochs of training
        model_saving_interval -- The model will be saved to
                                 model_saving_file_path after this many epochs

        Notes:
        The neural network architecture is the same as used in the DQN NIPS
        paper https://arxiv.org/abs/1312.5602
        """
        self.env = env
        self.lr = lr
        self.rmsprop_rho = rmsprop_rho
        self.minibatch_size = minibatch_size
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
                Conv2D(filters = 16, kernel_size = 8, strides = 4,
                       input_shape = (4, 84, 84),
                       data_format = "channels_first",
                       activation = "relu",
                       )
                )
            self.model.add(
                Conv2D(filters = 32, kernel_size = 4, strides = 2,
                       data_format = "channels_first", activation = "relu",
                       )
                )
            self.model.add(Flatten())
            self.model.add(
                Dense(units = 256, activation = "relu")
                )
            self.model.add(
                Dense(units = self.env.action_space.n,
                     activation = "linear",
                     )
                )

            optimizer = RMSprop(lr = self.lr, rho = self.rmsprop_rho)

            self.model.compile(loss = "mse", optimizer = optimizer)

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

    def get_feature_array_from_observation(self, observation):
        """Get feature from observation

        Arguments:
        observation -- Gym observation

        Notes:
        In the case of Atari environments, we wrap them so that they usually
        return 4 concatenated 84 x 84 grayscale images. The pixels have
        values from 0 to 255. We do this because the replay memory contains
        lots of observations and is held in the RAM. To save memory, we store
        pixels as integers since they take up less memory than floats.

        Before feeding this into the neural network, it is a good idea to scale
        the pixels and convert them to floats that lie between 0 and 1. Neural
        netowrks work better with floats. This is exactly what we do in this
        function.
        """

        feature_array = np.array(
            [
                observation[i] for i in range(len(observation))
                ]
            )

        feature_array = np.float32(feature_array/255.)

        return feature_array

    def update_model(self, discount_factor, replay_memory):
        """Update the weights of the model given the agent's experience using
        Q learning update (off policy)

        Arguments:
        discount_factor -- Quantifies how much the agent cares about future
                           rewards while learning. Often referred to as gamma in
                           the literature.
        replay_memory -- Replay memory is a deque containing experiences.
                         Experiences are of the form:
                         {
                            "observation" : observation,
                            "action" : action,
                            "reward" : reward,
                            "done" : done,
                            "next_observation" : next_observation
                            }
        """

        ## Choose a minibatch of experiences randomly from the replay memory.

        replay_memory_size = len(replay_memory)

        # We choose random indices in the replay memory. The experiences stored
        # in these indices is our minibatch.

        transition_indices = [
            random.randint(0, replay_memory_size - 1)
            for i in range(self.minibatch_size)
            ]

        ## Get current model predictions on the observations in minibatch

        model_input = np.array(
            [
                self.get_feature_array_from_observation(
                    replay_memory[transition_index]["observation"]
                    ) for transition_index in transition_indices
                ]
            )

        predictions = self.model.predict_on_batch(model_input)

        ## Get current model predictions on the next observation in minibatch

        next_model_input = np.array(
            [
                self.get_feature_array_from_observation(
                    replay_memory[transition_index]["next_observation"]
                    ) for transition_index in transition_indices
                ]
            )

        next_predictions = self.model.predict_on_batch(next_model_input)

        ## Get Q learning targets

        targets = []
        for i in range(len(transition_indices)):
            transition_index = transition_indices[i]
            action = replay_memory[transition_index]["action"]
            done = replay_memory[transition_index]["done"]
            reward = replay_memory[transition_index]["reward"]

            target = reward
            if not done:
                next_q_value = np.max(next_predictions[i])
                target += discount_factor*next_q_value

            targets_for_this_transition = predictions[i]
            targets_for_this_transition[action] = target

            targets.append(targets_for_this_transition)

        targets = np.asarray(targets)

        ## Update model

        self.model.fit(
            model_input,
            targets,
            batch_size = self.minibatch_size,
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
        model_input = np.array(
            [
                self.get_feature_array_from_observation(observation)
                ]
            )
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
