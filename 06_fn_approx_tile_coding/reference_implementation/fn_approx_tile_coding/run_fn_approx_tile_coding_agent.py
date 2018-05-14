import datetime
import json
import os
from pathlib import Path
import random

import argparse
import gym
import numpy as np

from fn_approx_tile_coding.fn_approx_tile_coding_agent import FnApproxAgent

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
# MountainCar-v0 and other default hyperparameters have been chosen for
# this environment. You may need to change some hyperparameters to make it
# work with other envs.
parser.add_argument("--env", default = "MountainCar-v0")

# Quantifies how much the agent cares about future rewards while learning.
# Often referred to as gamma in the literature.
parser.add_argument("--discount_factor", type = float, default = 1)

# Learning rate. Commonly known as alpha in the literature. In this
# implementation, we linearly anneal learning_rate from start_learning_rate
# to end_learning_rate during training.
parser.add_argument("--start_learning_rate", type = float, default = 0.5)
parser.add_argument("--end_learning_rate", type = float, default = 0.1)

# Epsilon denotes how much the agent explores and it is the probability of
# taking a random action. The start epsilon value is the value of epsilon
# at the beginning of training. The end epsilon is the value of epsilon at
# the end of training. The value is annealed linearly between these two values
# during training, so that the agent becomes more and more greedy as it
# learns.
parser.add_argument("--start_epsilon", type = float, default = 0.0)
parser.add_argument("--end_epsilon", type = float, default = 0.0)


# In tile coding, we use a set of tilings to cover the observation space. The
# tilings are staggered with respect to each other. Each tiling consists of a
# number of tiles. For more information, see
# https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view,
# Page 217.
parser.add_argument("--number_of_tiles", type = int, default = 10)
parser.add_argument("--number_of_tilings", type = int, default = 8)

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

# Seed for Gym environment used for training
parser.add_argument("--learning_env_seed", type = int, default = 0)

# Seed for Gym environment used for testing
parser.add_argument("--testing_env_seed", type = int, default = 0)

# End seed related parameters

### This section deals with training related parameters

# Start training related parameters

# We train the agent till this many observations
parser.add_argument("--total_observations", type = int, default = 10**7)

# We test the agent after this many observations. We test the agent on
# 100 episodes and report the average score per episode.
parser.add_argument("--test_interval", type = int, default = 5*10**4)

# Number of episodes to test the agent in every testing round
parser.add_argument(
    "--total_number_of_testing_episodes", type = int, default = 100
    )

# End training related parameters

## This section deals with logging related parameters

# Start logging related parameters

# The directory where the agent should store training logs.
parser.add_argument("--log_directory_path", default = "./training_logs")

# We store the model weights after this many episodes
parser.add_argument("--weight_saving_interval", type = int, default = 1000)

# End logging related parameters

args = parser.parse_args()

if __name__ == "__main__":

    ## Ensure reproducibility

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

    # In order to ensure reproducibility, we have to violate PEP8 and import
    # the class using tile coding only after the random module has been
    # seeded.
    from fn_approx_tile_coding.model_and_policy import TileCodingLinearFunction

    ## Create the necessary directories for logging the training process.
    ## Here is the directory structure
    ## training_logs    # log_directory_path
    ##     - CartPole-v0    # env_directory_path
    ##         - 2018-04-17_16:36:36    # timestamp_directory_path
    ##             - Logs go here....
    ##             - parameters.json    # parameter_file_path
    ##             - gym_training_logs    # gym_training_logs_directory_path
    ##             - gym_testing_logs    # gym_testing_logs_directory_path
    ##             - weights.txt    # weight_saving_file_path
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

    # Filepath for storing model weights
    weights_file_path = timestamp_directory_path / "weights.txt"

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

    # Get the function approximation model to be used by the agent.
    # This is a pluggable component of this algorithm, so you are always welcome to
    # define and use your model, instead of using the default one. The default one
    # is just supplied for reference. To use your own, define it in
    # models_and_policy.py and import and use it here.
    function = TileCodingLinearFunction(
        env = learning_env,
        number_of_tiles = args.number_of_tiles,
        number_of_tilings = args.number_of_tilings,
        weights_file_path = str(weights_file_path),
        )

    agent = FnApproxAgent()


    ## Train the agent. Progress will be printed on the command line.
    agent.train(
        function = function,
        discount_factor = args.discount_factor,
        start_learning_rate = args.start_learning_rate,
        end_learning_rate = args.end_learning_rate,
        start_epsilon = args.start_epsilon,
        end_epsilon = args.end_epsilon,
        learning_env = learning_env,
        testing_env = testing_env,
        total_observations = args.total_observations,
        test_interval = args.test_interval,
        total_number_of_testing_episodes =
            args.total_number_of_testing_episodes,
        gym_training_logs_directory_path =
            str(gym_training_logs_directory_path),
        gym_testing_logs_directory_path = str(gym_testing_logs_directory_path),
        weight_saving_interval = args.weight_saving_interval
        )
