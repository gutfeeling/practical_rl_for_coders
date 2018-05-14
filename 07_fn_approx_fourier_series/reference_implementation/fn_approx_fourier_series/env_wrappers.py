from math import atan, pi

import gym
import numpy as np

class AcrobotCartesianToPolarWrapper(gym.Wrapper):
    """A wrapper that converts X-Y coordinates of the links of Acrobot-v1
    to angles made by the links with respect to the vertical

    Notes:
    In Acrobot-v1, the 1st to 4th observations are the X-Y coordinates of the
    links of double pendulum. It turns out that the problem can be solved
    if we knew the angles made by the links with respect to the vertical
    instead of the X-Y coordinates. This would also reduce the observation
    space dimension by 2.
    """

    def __init__(self, env):
        """
        Arguments:
        env -- Gym environment to wrap
        """

        super(AcrobotCartesianToPolarWrapper, self).__init__(env)

        ## Modify the upper and lower bounds of the observation space
        ## to match the conversion from Cartesian to polar coordinates
        observation_upper_bounds = np.array(
            [
                pi, pi, self.env.observation_space.high[4],
                self.env.observation_space.high[5],
                ]
            )

        observation_lower_bounds = np.array(
            [
                -pi, -pi, self.env.observation_space.low[4],
                self.env.observation_space.low[5],
                ]
            )

        self.env.observation_space.high = observation_upper_bounds
        self.env.observation_space.low = observation_lower_bounds

    def step(self, action):
        """Override the step function to return rounded observations

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)

        # Convert X-Y coordinates to angles
        polar_observation = np.array(
            [
                atan(observation[1]/observation[0]),
                atan(observation[3]/observation[2]),
                observation[4],
                observation[5]
                ]
            )

        # Notice how we are returning the modified observation instead of
        # the original one
        return polar_observation, reward, done, info

    def reset(self):
        """Override the reset method of the env to output discretized observation
        """

        observation = self.env.reset()
        polar_observation = np.array(
            [
                atan(observation[1]/observation[0]),
                atan(observation[3]/observation[2]),
                observation[4],
                observation[5]
                ]
            )

        return polar_observation
