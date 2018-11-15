# Module 7: Function Approximation (Fourier Series)

At the end of this module

1. You will able to list a number of ways to extract features from state-action pairs
1. You will use Fourier decomposition to extract features from state-action pairs
2. You will implement linear function approximation with Fourier features to solve the `Acrobot-v1` environment :tada:

## Plan

1. :movie_camera: Fourier Series Refresher
2. :pencil: **Exercise**: *Find the Fourier coefficients upto order 5 for a square wave.*
3. :movie_camera: Using Fourier Series in RL function approximation - features, weights, updates, learning rate
4. :pencil: **Exercise**: *Wrap the `Acrobot-v1` environment to reduce the observation space from 6 dimensional to 4 dimensional by
converting cartesian co-ordinates to polar co-ordinates.*
5. :pencil: **Exercise**: *Implement a method in `ModelAndPolicy` to extract fourier coefficients from the
observations of `Acrobot-v0`.*
6. :pencil: **Exercise**: *Modify `ModelAndPolicy` to include a instance variable for weights, an instance variable for
learning rate per feature and write a method that returns the Q value by using linear function approximation.*
7. :pencil: **Exercise**: *Modify the update rule in `ModelAndPolicy`.*

## References

1. [Reinforcement Learning: An Introduction, Second Edition, Section 9.5.2](https://drive.google.com/file/d/1opPSz5AZ_kVa1uWOdOiveNiBFiEOHjkG/view)
2. [Acrobot-v1 Environment Details](https://github.com/openai/gym/wiki/Leaderboard#acrobot-v1)

