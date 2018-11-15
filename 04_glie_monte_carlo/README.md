# Module 4: GLIE Monte Carlo

At the end of this module

1. You will implement your first RL algorithm (GLIE Monte Carlo) to solve the `CartPole-v0` environment
2. You will be able to list a set of best practices when writing RL code in Python
3. You will be able to explain why an agent needs to simultaneously explore and exploit to solve an environment
4. You will be able to integrate epsilon-greedy exploration/exploitation schemes in your code
5. You will be able to write code that makes your results demonstrably reproducible

## Plan

1. :movie_camera: The Monte Carlo Update
2. :movie_camera: How to Structure RL Code
3. :pencil: **Exercise**: *Write an `Agent` class which will collect experiences and pass it on to a `ModelAndPolicy` class.
The `ModelAndPolicy` class should model the environments and implement a policy. Continue using a 
random policy for this exercise.*
4. :pencil: **Exercise**: *Modify `ModelAndPolicy` to use a table for modeling the environment. Update the model using Monte
Carlo updates.*
5. :movie_camera: Exploration vs. Exploitation - epsilon greedy policies and epsilon annealing
6. :pencil: **Exercise**: *Modify the `Agent` class to create an epsilon schedule during training.*
7. :pencil: **Exercise**: *Modify the `ModelAndPolicy` class to implement an epsilon greedy policy.*
8. :movie_camera: Creating the hyperparameter file
9. :movie_camera: Importance of reproducibility in RL
10. :pencil: **Exercise**: *Add command line options related to reproducibility in the hyperparamete file. Use `argparse`. Run
the agent twice and check if results are reproducible.

# References

1. [RL Course by David Silver, Lecture 5: Model Free Control (Till 38:46)](https://www.youtube.com/watch?v=lfHX2hHRMVQ&t)
2. [Coding Style Guide](https://github.com/gutfeeling/practical_rl_for_coders/blob/master/style_guide.md)
3. [CartPole-v0 Environment Details](https://github.com/openai/gym/wiki/Leaderboard#cartpole-v0)
