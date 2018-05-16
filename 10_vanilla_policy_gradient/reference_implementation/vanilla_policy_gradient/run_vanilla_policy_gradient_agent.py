import datetime
import json
from math import log, sqrt
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

# ActorCritic from vanilla_policy_gradient.model_and_policy will be imported
# later inside the executable block to ensure reproducibility. This file
# is the only place where we will be departing from PEP8 guidelines for
# importing all modules at the top of the file as we cannot ensure
# reproducibility if we stick to it.
from vanilla_policy_gradient.vanilla_policy_gradient_agent import (
    VanillaPolicyGradientAgent
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

# Start hypeparameters

# The name of the Gym environment to solve. The reference implementation assumes
# LunarLander-v2, and other default hyperparameters have been chosen for
# this environment. You may need to change some hyperparameters to make it
# work with other envs.
parser.add_argument("--env", default = "LunarLander-v2")

# Learning rate of the actor
parser.add_argument("--actor_learning_rate", type = float, default = 0.001)

# Learning rate of the critic
parser.add_argument("--lr_critic", type = float, default = 0.001)

# Quantifies how much the agent cares about future rewards while learning.
# Often referred to as gamma in the literature.
parser.add_argument("--discount_factor", type = float, default = 0.99)

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
parser.add_argument("--total_observations", type = int, default = 5e6)

# We test the agent after this many observations. We test the agent on
# 100 episodes and report the average score per episode.
parser.add_argument("--test_interval", type = int, default = 5e4)

# Number of episodes to test the agent in every testing round
parser.add_argument(
    "--total_number_of_testing_episodes", type = int, default = 100
    )

# End training related parameters

## This section deals with logging related parameters

# Start logging related parameters

# The directory where the agent should store training logs.
parser.add_argument("--log_directory_path", default = "./training_logs")

# After how many training epochs should we save the critic model
parser.add_argument("--critic_model_saving_interval", type = int, default = 200)

# After how many episdoes should we save the actor weights
parser.add_argument("--actor_weights_saving_interval", type = int, default = 200)

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
    from vanilla_policy_gradient.model_and_policy import (
        ActorCritic
        )

    ## Create the necessary directories for logging the training process.
    ## Here is the directory structure
    ## training_logs    # log_directory_path
    ##     - BipedalWalker-v2    # env_directory_path
    ##         - 2018-04-17_16:36:36    # timestamp_directory_path
    ##             - Logs go here....
    ##             - parameters.json    # parameter_file_path
    ##             - gym_training_logs    # gym_training_logs_directory_path
    ##             - gym_testing_logs    # gym_testing_logs_directory_path
    ##             - keras_training_logs    # keras_training_logs_directory_path
    ##             - actor_weights.txt   # actor_weights_saving_path
    ##             - critic_model.h5   # critic_model_saving_path
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

    # Directory for storing the loss of the critic
    keras_training_logs_directory_path = (
        timestamp_directory_path / "keras_training_logs"
        )

    keras_training_logs_directory_path.mkdir()

    critic_training_logs_file_path = (
        keras_training_logs_directory_path / "critic_logs.csv"
        )

    # Filepath for storing the actor weights and critic models after
    # interval specified by actor_weights_saving_interval and
    # critic_model_saving_interval
    actor_weights_saving_file_path = (
        timestamp_directory_path / "actor_weights.txt"
        )
    critic_model_saving_file_path = timestamp_directory_path / "critic_model.h5"

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

    # Get the learning env.
    learning_env = gym.make(args.env)

    if args.make_reproducible:
        learning_env.seed(args.learning_env_seed)

    # Get the testing env.
    testing_env = gym.make(args.env)

    if args.make_reproducible:
        testing_env.seed(args.testing_env_seed)

    # Get the actor critic model to be used by the agent.
    # This is a pluggable component of this algorithm, so you are always welcome to
    # define and use your model, instead of using the default one. The default one
    # is just supplied for reference. To use your own, define it in
    # models_and_policy.py and import and use it here.
    function = ActorCritic(
        env = learning_env,
        lr_critic = args.lr_critic,
        critic_training_logs_file_path = str(critic_training_logs_file_path),
        actor_weights_saving_file_path = str(actor_weights_saving_file_path),
        critic_model_saving_file_path = str(critic_model_saving_file_path),
        critic_model_saving_interval = args.critic_model_saving_interval,
        )

    agent = VanillaPolicyGradientAgent()


    ## Train the agent. Progress will be printed on the command line.
    agent.train(
        function = function,
        discount_factor = args.discount_factor,
        actor_learning_rate = args.actor_learning_rate,
        learning_env = learning_env,
        testing_env = testing_env,
        total_observations = args.total_observations,
        test_interval = args.test_interval,
        total_number_of_testing_episodes =
            args.total_number_of_testing_episodes,
        gym_training_logs_directory_path =
            str(gym_training_logs_directory_path),
        gym_testing_logs_directory_path = str(gym_testing_logs_directory_path),
        actor_weights_saving_interval = args.actor_weights_saving_interval
        )
