# Module 2: Reinforcement Learning Basics with OpenAI Gym

At the end of this module

1. You will be able to explain the concept of Agent, Environment, Action and Rewards. These are the basic
building blocks of RL
2. You will be able to install `OpenAI Gym`, instantiate environments, and control agents using Python
3. You will be able to explain the output returned by the `env.step(action)` method of `OpenAI Gym`
4. You will be able to define the learning goal of the agent in terms of maximizing the discounted reward sum
5. You will be able to able to modify `OpenAI Gym` environments, run simple training and testing loops, and 
log/monitor the agent's action
5. You will be able to decide whether RL or supervised learning is more suited to a given real world problem

## Plan

1. :movie_camera: What is `OpenAI Gym`?
2. :movie_camera: `OpenAI Gym` installation
3. :movie_camera: Agent, Environment, State, Action, Rewards
4. :movie_camera: Policy and discount rate
5. :pencil: **Exercise**: *Implement a training and testing loop for a random policy in the CartPole-v0 environment.*
6. :movie_camera: The importance of baselines
7. :movie_camera: `OpenAI Gym` environment wrappers
8. :pencil: **Exercise**: *Wrap the `CartPole-v0` environment to change the reward signature. The agent should get -1 reward if the
pole angle exceeds 15 degrees and 0 otherwise.*
9. :movie_camera: Monitoring using `OpenAI Gym`
10. :pencil: **Exercise**: *Implement automatic monitoring for the training and testing loops you wrote earlier.*
11. :movie_camera: Supervised learning vs. RL - when to use which?

## References

1. [RL Course by David Silver, Lecture 1: Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0)
2. [Getting Started with Gym](https://gym.openai.com/docs/#getting-started-with-gym)
3. [Wrappers](https://github.com/openai/gym/blob/master/gym/wrappers/README.md)
4. [Wrapper Example : The TimeLimit Class](https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py)
5. [Gym Monitoring (Tip : Study the public methods of the Monitor class, and read all docstrings)](https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py)
