import gym

class RewardScalingWrapper(gym.Wrapper):
    """A wrapper that scales rewards in Gym environment

    Notes:
    If the rewards have high variance, the neural network loss becomes
    unstable during training, causing large and catastrophic updates of
    the critic. A few outlier rewards is sufficient to cause trouble.
    This has been observed in (at least) the BipedalWalker-v2 environment.
    The simplest solution is to scale the rewards by multiplying it with a
    small number. This effectively reduces variance and leads to stable
    training of the critic.

    You may write your own wrapper that modifies the environment in
    a different way for your own experiments. Define the wrapper(s) in
    this file and then just import and use it/them in run_ppo_agent.py
    instead of this wrapper.
    """

    def __init__(self, env, reward_scaling_factor):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
        """

        super(RewardScalingWrapper, self).__init__(env)
        self.reward_scaling_factor = reward_scaling_factor

    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)
        scaled_reward = self.reward_scaling_factor*reward

        # notice that we are returning the scaled reward instead of the
        # original reward
        return observation, scaled_reward, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()
