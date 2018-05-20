from collections import deque

import cv2
import gym
import numpy as np


class PongReducedActionSpaceWrapper(gym.Wrapper):

    def __init__(self, env):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
        """

        super(PongReducedActionSpaceWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(3)
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

    def __init__(self, env, reward_upper_bound, reward_lower_bound):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
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
        reward = np.clip(
            reward, self.reward_lower_bound, self.reward_upper_bound
            )

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


class AtariGrayedResizedAndCroppedObservationWrapper(gym.Wrapper):

    def __init__(self, env):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
        """

        super(AtariGrayedResizedAndCroppedObservationWrapper, self).__init__(
            env
            )

    def gray_resize_and_crop_observation(self, observation):

        grayed_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_grayed_observation = cv2.resize(
            grayed_observation, (84, 110), interpolation=cv2.INTER_LINEAR
            )
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
        grayed_resized_and_cropped_observation = (
            self.gray_resize_and_crop_observation(observation)
            )

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

        grayed_resized_and_cropped_observation = (
            self.gray_resize_and_crop_observation(observation)
            )

        return grayed_resized_and_cropped_observation


class AtariObservationSkippingWrapper(gym.Wrapper):

    def __init__(self, env, frames_to_skip):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
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

        cumulative_reward_in_skipped_frames = 0

        for i in range(self.frames_to_skip):
            observation, reward, done, info = self.env.step(action)
            cumulative_reward_in_skipped_frames += reward
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

    def __init__(self, env, number_of_frames_to_concatenate):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
        """

        super(AtariFrameConcatenatingWrapper, self).__init__(env)
        self.number_of_frames_to_concatenate = number_of_frames_to_concatenate
        self.deque = deque([], maxlen = self.number_of_frames_to_concatenate)

    def step(self, action):
        """Override the step function to return scaled rewards

        Note:
        Never change the signature of the step function. It should always
        take an action as the only argument, and return observation, done,
        reward and info.
        """

        observation, reward, done, info = self.env.step(action)
        self.deque.append(observation)

        concatenated_observation = list(self.deque)

        return concatenated_observation, reward, done, info

    def reset(self):
        """
        Note:
        Gym requires wrappers to define a reset method, whether you want
        to modify the vanilla environment's reset method or not. If you don't,
        it throws a warning. The only purpose of this function is to avoid that
        warning.
        """

        observation = self.env.reset()
        self.deque.append(observation)

        for i in range(self.number_of_frames_to_concatenate - 1):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            self.deque.append(observation)

        concatenated_observation = list(self.deque)

        return concatenated_observation


class PongNIPSLearningEnvWrapper(gym.Wrapper):

    def __init__(self, env, reward_upper_bound, reward_lower_bound,
                 frames_to_skip, number_of_frames_to_concatenate
                 ):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
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

    def __init__(self, env, frames_to_skip, number_of_frames_to_concatenate
                 ):
        """
        Arguments:
        reward_scaling_factor -- Scale rewards by this number
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
