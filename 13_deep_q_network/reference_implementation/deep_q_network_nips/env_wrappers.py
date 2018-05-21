from collections import deque

import cv2
import gym
import numpy as np


class PongReducedActionSpaceWrapper(gym.Wrapper):
    """Reduce the action space for the Pong* environments

    Note:
    The OpenAI Gym version for Pong defines 6 actions, but only three
    are actually needed. These three actions are:

    0 -- Stay put
    2 -- Go up
    5 -- Go down

    We can reduce the complexity of the problem by reducing the action space
    to these three actions.
    """

    def __init__(self, env):
        """
        Arguments:
        env -- Gym environment to be wrapped
        """

        super(PongReducedActionSpaceWrapper, self).__init__(env)
        # Methods like env.action_space.sample() depend on action_space. We
        # need to modify the action space accordingly.
        self.action_space = gym.spaces.Discrete(3)

        # Mapping between the reduced action space to the full action space
        self.action_map = {
            0 : 0,
            1 : 2,
            2 : 5,
            }


    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        action = self.action_map[action]
        observation, reward, done, info = self.env.step(action)

        return observation, reward, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()


class AtariRewardClippingWrapper(gym.Wrapper):
    """Clip rewards in Atari environments

    Note:
    Training of the neural network is more stable if the rewards are scaled
    to be in the same order of magnitude. In some games, rewards may have
    large variations. We modify the reward functions such that rewards
    lie between two well defined values. This solves training instability.
    """

    def __init__(self, env, reward_upper_bound, reward_lower_bound):
        """
        Arguments:
        env -- Gym environment to be wrapped
        reward_upper_bound -- Rewards cannot go above this value
        reward_lower_bound -- Rewards cannot go below this value
        """

        super(AtariRewardClippingWrapper, self).__init__(env)
        self.reward_upper_bound = reward_upper_bound
        self.reward_lower_bound = reward_lower_bound

    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)
        # Clip rewards
        clipped_reward = np.clip(
            reward, self.reward_lower_bound, self.reward_upper_bound
            )

        # Notice how we are returning the clipped reward instead of the
        # original one
        return observation, clipped_reward, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()


class AtariGrayedResizedAndCroppedObservationWrapper(gym.Wrapper):
    """Turn the Atari emulator images into grayscale. Resize and crop it.

    The images returned by the Pong environment in Gym are RGB images with
    resolution 210 x 160. It turns out that the agent can learn well with
    a corresponding grayscale image. Also, 210 x 160 is too many pixels. It
    makes learning more difficult. So we resize the image to 84 x 84 to reduce
    the information complexity. However we do not scale down the 210 x 160
    image directly to the 84 x 84 image. We first scale it down to an 84 x 110
    image. Then we crop the image, choosing the playing area. This reduces
    the image to 84 x 84. This is the preprocessing that was done by the
    Google Deepmind group.
    """

    def __init__(self, env):
        """
        Arguments:
        env -- Gym environment to be wrapped
        """

        super(AtariGrayedResizedAndCroppedObservationWrapper, self).__init__(
            env
            )

    def gray_resize_and_crop_observation(self, observation):
        """Turn the 210 x 160 RGB image into a grayscale 84 x 84 image

        Arguments:
        observation -- Gym observation for Atari environment, which is a
                       numpy array of shape (210, 160, 3)
        """

        # Turn to grayscale
        grayed_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Resize to 110 x 84
        resized_grayed_observation = cv2.resize(
            grayed_observation, (84, 110), interpolation=cv2.INTER_LINEAR
            )
        # Crop to 84 x 84
        cropped_observation = resized_grayed_observation[17:101, :]

        return cropped_observation

    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)
        # Transform the 210 x 160 RGB image to 84 x 84 grayscale image
        grayed_resized_and_cropped_observation = (
            self.gray_resize_and_crop_observation(observation)
            )

        # Notice how we are returning the modified image instead of the
        # original one
        return grayed_resized_and_cropped_observation, reward, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        observation = self.env.reset()

        # Transform the 210 x 160 RGB image to 84 x 84 grayscale image
        grayed_resized_and_cropped_observation = (
            self.gray_resize_and_crop_observation(observation)
            )

        # Notice how we are returning the modified image instead of the
        # original one
        return grayed_resized_and_cropped_observation


class AtariObservationSkippingWrapper(gym.Wrapper):
    """Repeat the same action for a defined number of frames (n frames), and
    return the nth frame as observation

    Notes:
    To make learning faster, we can ignore the intermediate frames. The
    Google Deepmind team chose to consider every 4th observation. The agent
    makes a decision every 4th observation. This approximately matches human
    speed of decision making. Note that for the SpaceInvaders, you need to
    use every 3rd observation (instead of 4) because of a glitch in the emulator.

    The reward is the sum of the reward that the agent receives in the
    skipped observations.
    """

    def __init__(self, env, frames_to_skip):
        """
        Arguments:
        env -- Gym environment to be wrapped
        frames_to_skip -- Each frames_to_skipth observation is considered. The
                          intermediate observations are thrown away.
        """

        super(AtariObservationSkippingWrapper, self).__init__(env)
        self.frames_to_skip = frames_to_skip

    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        # The modified reward is the sum of the rewards in the skipped
        # observations
        cumulative_reward_in_skipped_frames = 0

        # Observation skipping
        for i in range(self.frames_to_skip):
            observation, reward, done, info = self.env.step(action)
            # Add up rewards in skipped observations
            cumulative_reward_in_skipped_frames += reward
            # If the episode ends while observation skipping, terminate
            if done:
                break

        return observation, cumulative_reward_in_skipped_frames, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()


class AtariFrameConcatenatingWrapper(gym.Wrapper):
    """Concatenate a defined number of observations, and return the
    concatenated version as the observation

    Notes:
    Passing a single observation doesn't work because it is a still picture.
    The agent cannot figure out dynamic information such as velocity of the
    ball, paddles, or other objects based on just one observation. To include
    dynamical information, we concatenate the last n observations and define
    that as the observation that the agent sees.

    Google Deepmind chose to concatenate 4 observations together.
    """

    def __init__(self, env, number_of_frames_to_concatenate):
        """
        Arguments:
        env --  Gym environment to be wrapped
        number_of_frames_to_concatenate -- Concatenate this many frames
        """

        super(AtariFrameConcatenatingWrapper, self).__init__(env)
        self.number_of_frames_to_concatenate = number_of_frames_to_concatenate

        # We use a deque to store the last number_of_frames_to_concatenate
        # observations.
        self.deque = deque([], maxlen = self.number_of_frames_to_concatenate)

    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)
        # When we add the observation to the deque, Python simply adds a
        # pointer in the deque pointing to the original memory location
        # where observation is stored. No new memory is needed for this
        # operation.
        self.deque.append(observation)

        # Return a list, not a deque
        # When we convert the deque to a list, Python simply adds a
        # pointer in the list pointing to the original memory location
        # where observation is stored. No new memory is needed for this
        # operation.
        concatenated_observation = list(self.deque)

        return concatenated_observation, reward, done, info

    def reset(self):
        """
        Note:
        To ensure that we have always have at least
        number_of_frames_to_concatenate observations in memory, we let
        the agent take number_of_frames_to_concatenate actions in the
        environment reset and accumulate the necessary history right at the
        beginning of an episode.
        """

        observation = self.env.reset()
        self.deque.append(observation)

        # Take number_of_frames_to_concatenate actions and store observations
        # in memory
        for i in range(self.number_of_frames_to_concatenate - 1):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            self.deque.append(observation)

        # Return a list, not a deque.
        concatenated_observation = list(self.deque)

        return concatenated_observation


class PongNIPSLearningEnvWrapper(gym.Wrapper):
    """The wrapper for the Pong learning environment. This puts together
    all the other wrappers needed for the learing environment to create an
    unified wrapper.

    Notes:
    We make the following changes to the original Gym environment in the
    following order:
    1. Reduce the action space of Pong from 6 to 3.
    2. Clip rewards
    3. Turn the 210 x 160 RGB observations to 84 x 84 grayscale observations
    4. Implement frame skipping
    5. Concatenate a defined number of frames to include dynamical information
    """

    def __init__(self, env, reward_upper_bound, reward_lower_bound,
                 frames_to_skip, number_of_frames_to_concatenate
                 ):
        """
        Arguments:
        env --  Gym environment to be wrapped
        reward_upper_bound -- Rewards cannot go above this value
        reward_lower_bound -- Rewards cannot go below this value
        frames_to_skip -- Each frames_to_skipth observation is considered. The
                          intermediate observations are thrown away.
        number_of_frames_to_concatenate -- Concatenate this many frames
        """

        env = AtariFrameConcatenatingWrapper(
            AtariObservationSkippingWrapper(
                AtariGrayedResizedAndCroppedObservationWrapper(
                    AtariRewardClippingWrapper(
                        PongReducedActionSpaceWrapper(
                            env
                        ),
                        reward_upper_bound = reward_upper_bound,
                        reward_lower_bound = reward_lower_bound,
                    ),
                ),
                frames_to_skip = frames_to_skip,
                ),
            number_of_frames_to_concatenate = number_of_frames_to_concatenate,
            )

        super(PongNIPSLearningEnvWrapper, self).__init__(env)

    def step(self, action):

        return self.env.step(action)

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()


class PongNIPSTestingEnvWrapper(gym.Wrapper):
    """The wrapper for the Pong testing environment. This puts together
    all the other wrappers needed for the testing environment to create an
    unified wrapper. Same as the learning environment wrapper except for the
    reward clipping part. We don't clip rewards in the testing environment
    because we want to benchmark with respect to the original game scores.

    Notes:
    We make the following changes to the original Gym environment in the
    following order:
    1. Reduce the action space of Pong from 6 to 3.
    2. Turn the 210 x 160 RGB observations to 84 x 84 grayscale observations
    3. Implement frame skipping
    4. Concatenate a defined number of frames to include dynamical information
    """

    def __init__(self, env, frames_to_skip, number_of_frames_to_concatenate
                 ):
        """
        Arguments:
        Arguments:
        env --  Gym environment to be wrapped
        frames_to_skip -- Each frames_to_skipth observation is considered. The
                          intermediate observations are thrown away.
        number_of_frames_to_concatenate -- Concatenate this many frames
        """

        env = AtariFrameConcatenatingWrapper(
            AtariObservationSkippingWrapper(
                AtariGrayedResizedAndCroppedObservationWrapper(
                    PongReducedActionSpaceWrapper(
                        env
                        ),
                    ),
                frames_to_skip = frames_to_skip,
                ),
            number_of_frames_to_concatenate = number_of_frames_to_concatenate,
            )

        super(PongNIPSTestingEnvWrapper, self).__init__(env)

    def step(self, action):

        return self.env.step(action)

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        return self.env.reset()
