# Module 11: Proximal Policy Optimization

At the end of this module

1. You will able to list the limitations of Vanilla Policy Gradient methods
2. You will be able to explain why Proximal Policy Optimization with GAE leads to better learning stability and sample efficiency
3. You will implement the state-of-the-art Proximal Policy Optimization algorithm with Generalized Advantage Estimation (GAE) to solve 
the `BipedalWalker-v2` environment :tada:
4. You will implement a custom `Keras` loss function

## Plan 

1. :movie_camera: Stability of learning
2. :movie_camera: Sample efficiency
3. :movie_camera: Motivation behind PPO updates and comparison to other methods
4. :movie_camera: PPO Loss
5. :pencil: **Exercise**: *Implement a Neural Network with the PPO loss in `ModelAndPolicy`.*
6. :movie_camera: Generalized Advantage Estimation (GAE)
7. :pencil: **Exercise**: *Modify the `Agent` class to return GAE experiences.*
8. :pencil: **Exercise**: *Implement the GAE update rule in `ModelAndPolicy`.*

## References

1. [Deep RL Bootcamp, Lecture 5: Natural Policy Gradients, TRPO, and PPO by John Schulman](https://www.youtube.com/watch?v=xvRrgxcpaHY)
2. [BipedalWalker-v2 environment details](https://github.com/openai/gym/wiki/Leaderboard#bipedalwalker-v2)

## Navigation 

[Next - Module 12: RL On `Google Cloud`](https://github.com/gutfeeling/practical_rl_for_coders/tree/master/12_rl_on_google_cloud)

[Previous - Module 10: Vanilla Policy Gradient](https://github.com/gutfeeling/practical_rl_for_coders/tree/master/10_vanilla_policy_gradient)

