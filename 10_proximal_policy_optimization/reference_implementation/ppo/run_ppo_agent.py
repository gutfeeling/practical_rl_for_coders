import datetime
import json
from math import log, sqrt
from pathlib import Path

import argparse
import gym

from ppo.env_wrappers import RewardScalingWrapper
from ppo.models import DefaultActor, DefaultCritic
from ppo.ppo_agent import PPOAgent

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
# BipedalWalker-v2, and other default hyperparameters have been chosen for
# this environment. You may need to change some hyperparameters to make it
# work with other envs.
parser.add_argument("--env", default = "BipedalWalker-v2")

# Sometimes, we want to modify the environment to make learning easier. We
# use Gym wrappers for this purpose. The following hyperparameter controls
# how the learning environment is modified. Based on this hyperparameter,
# we will wrap the vanilla env with the appropriate wrapper to create the
# learning environment. (The appropriate wrapper should be defined in
# env_wrappers.py and imported here). For the reference implementation, we
# use a wrapper that scales rewards by 1e-2 to make training more stable.
parser.add_argument("--learning_env_wrapper",
                    default = "reward_scaling_wrapper",
                    choices = ["no_wrapper", "reward_scaling_wrapper"]
                    )
parser.add_argument("--reward_scaling_factor", type = float,
                    default = 1e-2
                    )

# We may also want to wrap the environment where we test the agent's
# performance. The folowing hyperparamter controls how the testing env
# is wrapped. For the reference implementation, we simply use the
# vanilla Gym environment because we would like to compare our agent's
# performance with other implementations.
parser.add_argument("--testing_env_wrapper", default = "no_wrapper",
                    choices = ["no_wrapper", "reward_scaling_wrapper"]
                    )

# The reference implementation assumes that the policy is a multivariate
# Gaussian with constant spherical covariance. The constant variance is
# not learned, but rather becomes a hyperparameter in this implementation.
# You may want to use other models e.g. one where a diagonal covariance is
# learned as well. In this case, you need to define the appropriate actor
# in models.py and then import and use it here. If that model requires
# additional hyperparameters, then define additional command line arguments
# for them.
parser.add_argument("--var", type = float, default = sqrt(log(2)))

# Learning rate of the actor
parser.add_argument("--lr_actor", type = float, default = 3e-4)

# The loss clipping parameter in the PPO L_CLIP loss. See the original
# PPO paper, where this parameter is called epsilon.
parser.add_argument("--loss_clipping_epsilon", type = float,
                    default = 0.2
                    )

# Learning rate of the critic
parser.add_argument("--lr_critic", type = float, default = 3e-4)

# Quantifies how much the agent cares about future rewards while learning.
# Often referred to as gamma in the literature.
parser.add_argument("--discount_factor", type = float, default = 0.99)

# This is the lambda in TD(lambda). We use a finite horizon version of
# TD(lambda) to estimate advantages and value targets, as described in the
# original PPO paper.
parser.add_argument("--lambda_value", type = float, default = 0.95)

# Number of experiences to collect before performing a training step. Must be a
# integer multiple of minibatch_size. This is also an upper bound of how
# much the agent looks into the future to compute advantages and value targets.
parser.add_argument("--horizon", type = int, default = 2048)

# Minibatch size
parser.add_argument("--minibatch_size", type = int, default = 64)

# Number of epochs of training for a given set of experiences
parser.add_argument("--epochs", type = int, default = 10)

# End hyperparameters

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

# After how many training epochs should we save the actor model
parser.add_argument("--actor_model_saving_interval", type = int, default = 200)

# After how many training epochs should we save the critic model
parser.add_argument("--critic_model_saving_interval", type = int, default = 200)

# End logging related parameters

args = parser.parse_args()

if __name__ == "__main__":

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
    ##             - actor_model.h5    # actor_model_saving_path
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

    # Directory for storing the loss of the actor and the critic
    keras_training_logs_directory_path = (
        timestamp_directory_path / "keras_training_logs"
        )

    keras_training_logs_directory_path.mkdir()

    actor_training_logs_file_path = (
        keras_training_logs_directory_path / "actor_logs.csv"
        )
    critic_training_logs_file_path = (
        keras_training_logs_directory_path / "critic_logs.csv"
        )

    # Filepath for storing the actor and critic models after interval specified
    # by actor_model_saving_interval and critic_model_saving_interval
    actor_model_saving_path = timestamp_directory_path / "actor_model.h5"
    critic_model_saving_path = timestamp_directory_path / "critic_model.h5"

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
    if args.learning_env_wrapper == "reward_scaling_wrapper":
        learning_env = RewardScalingWrapper(
            env = gym.make(args.env),
            reward_scaling_factor = args.reward_scaling_factor
            )
    else:
        learning_env = gym.make(args.env)

    # Get the testing env. Wrap the vanilla env with the appropriate wrapper
    # or leave it unmodified depending on the testing_env_wrapper
    # hyperparameter
    if args.testing_env_wrapper == "reward_scaling_wrapper":
        testing_env = RewardScalingWrapper(
            env = gym.make(args.env),
            reward_scaling_factor = args.reward_scaling_factor
            )
    else:
        testing_env = gym.make(args.env)

    # Get the actor. The actor is a pluggable component of this algorithm, so
    # you are always welcome to define and use your own actors, instead
    # of using the default one. The default one is just supplied for
    # reference. To use your own actor, define it in models.py
    # and import and use it here.
    actor = DefaultActor(
        env = learning_env,
        var = args.var,
        lr = args.lr_actor,
        loss_clipping_epsilon = args.loss_clipping_epsilon,
        training_logs_file_path = str(actor_training_logs_file_path),
        model_saving_path = str(actor_model_saving_path),
        model_saving_interval = args.actor_model_saving_interval
        )

    # Get the critic. The critic is a pluggable component of this algorithm, so
    # you are welcome to use your own critic too.
    critic = DefaultCritic(
        env = learning_env,
        lr = args.lr_critic,
        training_logs_file_path = str(critic_training_logs_file_path),
        model_saving_path = str(critic_model_saving_path),
        model_saving_interval = args.critic_model_saving_interval
        )

    agent = PPOAgent()

    ## Train the agent. Progress will be printed on the command line.
    agent.train(
        actor = actor,
        critic = critic,
        discount_factor = args.discount_factor,
        lambda_value = args.lambda_value,
        learning_env = learning_env,
        testing_env = testing_env,
        horizon = args.horizon,
        minibatch_size = args.minibatch_size,
        epochs = args.epochs,
        total_observations = args.total_observations,
        test_interval = args.test_interval,
        total_number_of_testing_episodes =
            args.total_number_of_testing_episodes,
        gym_training_logs_directory_path =
            str(gym_training_logs_directory_path),
        gym_testing_logs_directory_path = str(gym_testing_logs_directory_path)
        )
