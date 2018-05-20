import datetime
import json
import os
from pathlib import Path
import random

import argparse
import gym
# Keras will be imported later inside the executable block to ensure
# reproducibility. This file is the only place where we will be departing
# from PEP8 guidelines for importing all modules at the top of the file as we
# cannot ensure reproducibility if we stick to it.
import numpy as np
import tensorflow as tf

# NeuralNetwork from fn_approx_neural_network.model_and_policy will be imported
# later inside the executable block to ensure reproducibility. This file is the
# only place where we will be departing from PEP8 guidelines for importing all
# modules at the top of the file as we cannot ensure reproducibility if we stick
# to it.
from deep_q_network_nips.deep_q_network_nips_agent import DQNNIPSAgent
from deep_q_network_nips.env_wrappers import (
    PongNIPSLearningEnvWrapper, PongNIPSTestingEnvWrapper
    )

parser = argparse.ArgumentParser()

## This section deals with hyperparameters.
## Can be supplied as command line arguments to this script.
## The default values are the default hyperparameters for the reference
## implementation. They have been tested and known to work well. These
## default values have not been (and should not be) overriden in any other file
## (We want to have a single source of truth for hyperparameters - and it's
## this file!).
## The only way to change them is to provide your own values as command line
## arguments while running this script.

# Start hyperparameters

# The name of the Gym environment to solve. The reference implementation assumes
# CartPole-v0 and other default hyperparameters have been chosen for
# this environment. You may need to change some hyperparameters to make it
# work with other envs.
parser.add_argument("--env", default = "PongNoFrameskip-v4")

# Sometimes, we want to modify the environment to make learning easier. We
# use Gym wrappers for this purpose. The following hyperparameter controls
# how the learning environment is modified. Based on this hyperparameter,
# we will wrap the vanilla env with the appropriate wrapper to create the
# learning environment. (The appropriate wrapper should be defined in
# env_wrappers.py and imported here). For the reference implementation, we
# use a wrapper that implements the preprocessing used in the NIPS paper and
# reduces the action space.
parser.add_argument("--learning_env_wrapper",
                    default = "pong_nips_learning_env_wrapper",
                    choices = [
                        "no_wrapper",
                        "pong_nips_learning_env_wrapper",
                        "pong_nips_testing_env_wrapper"
                        ]
                    )

parser.add_argument("--reward_upper_bound", type = float, default = 1.0)

parser.add_argument("--reward_lower_bound", type = float, default = -1.0)

parser.add_argument("--frames_to_skip", type = int, default = 4)

parser.add_argument("--number_of_frames_to_concatenate",
                    type = int, default = 4
                    )

# We may also want to wrap the environment where we test the agent's
# performance. The folowing hyperparamter controls how the testing env
# is wrapped. For the reference implementation, we simply use almost the
# same wrapper as used for the learning env, but without the reward clipping.

parser.add_argument("--testing_env_wrapper",
                    default = "pong_nips_testing_env_wrapper",
                    choices = [
                        "no_wrapper",
                        "pong_nips_learning_env_wrapper",
                        "pong_nips_testing_env_wrapper"
                        ]
                    )

# Quantifies how much the agent cares about future rewards while learning.
# Often referred to as gamma in the literature.
parser.add_argument("--discount_factor", type = float, default = 0.95)

# Learning rate of the neural network
parser.add_argument("--lr", type = float, default = 0.0002)

parser.add_argument("--rmsprop_rho", type = float, default = 0.99)

# Epsilon denotes how much the agent explores and it is the probability of
# taking a random action. The start epsilon value is the value of epsilon
# at the beginning of training. The end epsilon is the value of epsilon at
# the end of training. The value is annealed linearly between these two values
# during training, so that the agent becomes more and more greedy as it
# learns.
parser.add_argument("--start_epsilon", type = float, default = 1)
parser.add_argument("--end_epsilon", type = float, default = 0.1)

parser.add_argument("--replay_memory_size", type = int, default = 10**6)

parser.add_argument("--minibatch_size", type = int, default = 32)

# End hyperparameters

### This section deals with reproducibility. Note the reproducibility comes
### with a performance penalty as we cannot use Tensorflow's default
### multithreading anyone - it introduces randomness that cannot be
### reproduced perfectly.

# Begin reproducibility related parameters

# Whether you want reproducible result. If this is False, then the seed
# related arguments are ignored.
parser.add_argument("--make_reproducible", type= bool, default = False)

# Seed for numpy.random
parser.add_argument("--numpy_seed", type = int, default = 0)

# Seed for Python STDLIB random module
parser.add_argument("--random_seed", type = int, default = 0)

# Seed for PYTHONHASHSEED
parser.add_argument("--python_hash_seed", type = int, default = 0)

# Seed for Tensorflow
parser.add_argument("--tensorflow_seed", type = int, default = 0)

# Seed for Gym environment used for training
parser.add_argument("--learning_env_seed", type = int, default = 0)

# Seed for Gym environment used for testing
parser.add_argument("--testing_env_seed", type = int, default = 0)

# End seed related parameters

### This section deals with training related parameters

# Start training related parameters

# We train the agent till this many observations
parser.add_argument("--total_observations", type = int, default = 5*10**6)

# We test the agent after this many observations. We test the agent on
# 100 episodes and report the average score per episode.
parser.add_argument("--test_interval", type = int, default = 5*10**4)

# Number of episodes to test the agent in every testing round
parser.add_argument(
    "--total_number_of_testing_episodes", type = int, default = 100
    )

parser.add_argument(
    "--observation_number_when_epsilon_annealing_ends", type = int,
    default = 10**6
    )

parser.add_argument(
    "--observation_number_when_training_starts", type = int, default = 100
    )

# End training related parameters

## This section deals with logging related parameters

# Start logging related parameters

# The directory where the agent should store training logs.
parser.add_argument("--log_directory_path", default = "./training_logs")

# We store the Keras model after this many episodes
parser.add_argument("--model_saving_interval", type = int, default = 5*10**4)

# End logging related parameters

args = parser.parse_args()

if __name__ == "__main__":

    ## Ensure reproducibility
    ## See the following section in https://keras.io/getting-started/faq/
    ## Section : How can I obtain reproducible results using Keras during
    ## development?

    if args.make_reproducible:

        # The below is necessary in Python 3.2.3 onwards to
        # have reproducible behavior for certain hash-based operations.
        # See these references for further details:
        # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
        # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

        os.environ["PYTHONHASHSEED"] = str(args.python_hash_seed)

        # The below is necessary for starting Numpy generated random numbers
        # in a well-defined initial state.

        np.random.seed(args.numpy_seed)

        # The below is necessary for starting core Python generated random
        # numbers in a well-defined state.

        random.seed(args.random_seed)

        # Force TensorFlow to use single thread.
        # Multiple threads are a potential source of
        # non-reproducible results.
        # For further details, see:
        # https://stackoverflow.com/questions/42022950/

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
            )

        # Keras must be imported after setting numpy.random and random seeds.
        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see:
        # https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        tf.set_random_seed(args.tensorflow_seed)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    # Contains Keras related imports, so must be imported after everything
    # to ensure reproducibility
    from deep_q_network_nips.model_and_policy import NeuralNetwork

    ## Create the necessary directories for logging the training process.
    ## Here is the directory structure
    ## training_logs    # log_directory_path
    ##     - CartPole-v0    # env_directory_path
    ##         - 2018-04-17_16:36:36    # timestamp_directory_path
    ##             - Logs go here....
    ##             - parameters.json    # parameter_file_path
    ##             - gym_training_logs    # gym_training_logs_directory_path
    ##             - gym_testing_logs    # gym_testing_logs_directory_path
    ##             - keras_training_logs    # keras_training_logs_directory_path
    ##             - model.h5    # model_saving_file_path

    log_directory_path = Path(args.log_directory_path)

    if not log_directory_path.exists():
        log_directory_path.mkdir()

    env_directory_path = log_directory_path / args.env

    if not env_directory_path.exists():
        env_directory_path.mkdir()

    # All data for this training run will be stored under a directory that is
    # named after the start time of the training run
    timestamp_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    timestamp_directory_path = env_directory_path / timestamp_string

    timestamp_directory_path.mkdir()

    # Directories for storing automatic gym logs for training and testing
    # This includes rewards per training episode and videos of test episodes.
    gym_training_logs_directory_path = (
        timestamp_directory_path / "gym_training_logs"
        )

    gym_testing_logs_directory_path = (
        timestamp_directory_path / "gym_testing_logs"
        )

    # Directory for storing the loss of the neural network
    keras_training_logs_directory_path = (
        timestamp_directory_path / "keras_training_logs"
        )

    keras_training_logs_directory_path.mkdir()

    training_logs_file_path = (
        keras_training_logs_directory_path / "logs.csv"
        )

    # Filepath for storing models after interval specified
    # by model_saving_interval
    model_saving_file_path = timestamp_directory_path / "model.h5"

    # Filepath for storing all parameters for this run
    parameters_file_path = timestamp_directory_path / "parameters.json"


    ## Pre training tasks

    # Save parameters before starting training. Useful for future reference and
    # for reproduction of results.

    with parameters_file_path.open("w") as parameters_fh:
        # Get a dictionary of parameters including hyperparameters,
        # training related parameters and logging related parameters
        parameters = vars(args)
        json.dump(parameters, parameters_fh)

    # Get the learning env. Wrap the vanilla env with the appropriate wrapper
    # or leave it unmodified depending on the learning_env_wrapper
    # hyperparameter
    if args.learning_env_wrapper == "pong_nips_learning_env_wrapper":
        learning_env = PongNIPSLearningEnvWrapper(
            env = gym.make(args.env),
            reward_upper_bound = args.reward_upper_bound,
            reward_lower_bound = args.reward_lower_bound,
            frames_to_skip = args.frames_to_skip,
            number_of_frames_to_concatenate =
                args.number_of_frames_to_concatenate,
            )
    elif args.learning_env_wrapper == "pong_nips_testing_env_wrapper":
        learning_env = PongNIPSTestingEnvWrapper(
            env = gym.make(args.env),
            frames_to_skip = args.frames_to_skip,
            number_of_frames_to_concatenate =
                args.number_of_frames_to_concatenate,
            )
    else:
        learning_env = gym.make(args.env)

    if args.make_reproducible:
        learning_env.seed(args.learning_env_seed)

    # Get the testing env. Wrap the vanilla env with the appropriate wrapper
    # or leave it unmodified depending on the testing_env_wrapper
    # hyperparameter
    if args.testing_env_wrapper == "pong_nips_learning_env_wrapper":
        testing_env = PongNIPSLearningEnvWrapper(
            env = gym.make(args.env),
            reward_upper_bound = args.reward_upper_bound,
            reward_lower_bound = args.reward_lower_bound,
            frames_to_skip = args.frames_to_skip,
            number_of_frames_to_concatenate =
                args.number_of_frames_to_concatenate,
            )
    elif args.testing_env_wrapper == "pong_nips_testing_env_wrapper":
        testing_env = PongNIPSTestingEnvWrapper(
            env = gym.make(args.env),
            frames_to_skip = args.frames_to_skip,
            number_of_frames_to_concatenate =
                args.number_of_frames_to_concatenate,
            )
    else:
        testing_env = gym.make(args.env)

    if args.make_reproducible:
        testing_env.seed(args.testing_env_seed)

    # Get the function approximation model to be used by the agent.
    # This is a pluggable component of this algorithm, so you are always welcome to
    # define and use your model, instead of using the default one. The default one
    # is just supplied for reference. To use your own, define it in
    # model_and_policy.py and import and use it here.
    function = NeuralNetwork(
        env = learning_env,
        lr = args.lr,
        rmsprop_rho = args.rmsprop_rho,
        minibatch_size = args.minibatch_size,
        model_saving_file_path = str(model_saving_file_path),
        model_saving_interval = args.model_saving_interval,
        training_logs_file_path = str(training_logs_file_path)
        )

    agent = DQNNIPSAgent()


    ## Train the agent. Progress will be printed on the command line.
    agent.train(
        function = function,
        discount_factor = args.discount_factor,
        start_epsilon = args.start_epsilon,
        end_epsilon = args.end_epsilon,
        observation_number_when_epsilon_annealing_ends =
            args.observation_number_when_epsilon_annealing_ends,
        replay_memory_size = args.replay_memory_size,
        learning_env = learning_env,
        testing_env = testing_env,
        total_observations = args.total_observations,
        observation_number_when_training_starts =
            args.observation_number_when_training_starts,
        test_interval = args.test_interval,
        total_number_of_testing_episodes =
            args.total_number_of_testing_episodes,
        gym_training_logs_directory_path =
            str(gym_training_logs_directory_path),
        gym_testing_logs_directory_path = str(gym_testing_logs_directory_path),
        )
