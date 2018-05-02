import gym
import numpy as np

class ObservationRoundingWrapper(gym.Wrapper):
    """A wrapper that rounds observations to a certain number of decimal places

    Notes:
    The GLIE Monte Carlo method requires that we maintain a dictionary
    of Q values for any given state action pair. We should also ensure that
    the number of state action pairs is not too large, because then we would
    have to learn the Q value for all of them. This would take a long time.

    In environments like CartPole-v0, the observation space is continuous, while
    the action space is discrete and small. The number of state action pair is
    effectively infinite because of the continuous action space. We need to
    reduce the number of states in order to apply GLIE Monte Carlo to such
    problems.

    In practice, we know that the Q values of state action pairs that do not
    differ too much from each other must be close. Therefore, it makes sense
    to just put them in the same bin and learn just one Q value for them. This
    speeds up learning.

    A reasonable way to bin similar states is to consider two states that are
    the same upto a certain number of decimal places to be the same state.
    """

    def __init__(self, env, number_of_decimal_places):
        """
        Arguments:
        number_of_decimal_places -- Round observations to this many decimal
                                    places
        """

        super(ObservationRoundingWrapper, self).__init__(env)
        self.number_of_decimal_places = number_of_decimal_places

    def step(self, action):
        """Override the step function to return rounded observations

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)
        rounded_observation = np.round(
            observation, decimals = self.number_of_decimal_places,
            )
        # notice that we are returning the rounded observation instead of the
        # original observation
        return rounded_observation, reward, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()
