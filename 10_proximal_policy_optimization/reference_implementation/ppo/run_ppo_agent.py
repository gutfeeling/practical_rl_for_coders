from math import log, sqrt

import argparse
import gym

from ppo.env_wrappers import RewardScalingWrapper
from ppo.models import DefaultActor, DefaultCritic
from ppo.ppo_agent import PPOAgent

parser = argparse.ArgumentParser()

### This section deals with hyperparameters.
### Can be supplied as command line arguments to this script.

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


# Start training related parameters

# We train the agent till this many observations
parser.add_argument("--total_observations", type = int, default = 5e6)

# We test the agent after this many observations. We test the agent on
# 100 episodes and report the average score per episode.
parser.add_argument("--test_interval", type = float, default = 5e4)

# End training related parameters

args = parser.parse_args()

if __name__ == "__main__":

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

    # You are always welcome to define and use your own actors, instead
    # of using the default one. The default one is just supplied for
    # reference. To use your own actor, define it in models.py
    # and import and use it here.
    actor = DefaultActor(env = learning_env, var = args.var, lr = args.lr_actor,
                         loss_clipping_epsilon = args.loss_clipping_epsilon
                         )

    # You are welcome to use your own critic too.
    critic = DefaultCritic(env = learning_env, lr = args.lr_critic)

    agent = PPOAgent()

    # Train the agent. Progress will be printed on the command line.
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
        test_interval = args.test_interval
        )
