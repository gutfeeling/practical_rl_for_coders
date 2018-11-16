# Module 10: Vanilla Policy Gradient

At the end of this module:

1. You will be able to explain why value based methods don't work so well in environments with continuous action spaces
2. You will be able to incrementally improve the agent's policy directly without performing any value iterations
3. You will be able to define the learning goal in terms of finding a policy that maximizes the score function
4. You will be able to choose parameterized policy functions according to the nature of the action space
5. You will implement a simple Policy Gradient algorithm (called REINFORCE) that iterates a Softmax policy to solve the 
`LunarLander-v2` environment :tada:
6. You will be able to list limitations of REINFORCE and describe how Actor Critic architectures address these limitations
7. You will implement an Actor Critic architecture to solve the `LunarLander-v2` environment again :tada:

## Plan

1. :movie_camera: Limitations of using value based methods - continuous action spaces
2. :movie_camera: Policy iterations and the score function
3. :movie_camera: Discrete actions - Softmax policy
4. :movie_camera: Continuous action spaces - Gaussian policy
4. :pencil: **Exercise**: *In the `LunarLander-v2` environment, modify `ModelAndPolicy` to include a function that 
gets the features.*
4. :pencil: **Exercise**: *Replace epsilon greedy action with a softmax policy.*
5. :movie_camera: REINFORCE and its limitations
6. :movie_camera: Actor Critic architectures
7. :pencil: **Exercise**: *Implement a critic in `ModelAndPolicy` using a neural network.*
8. :pencil: **Exercise**: *Implement the actor critic update rule in `ModelAndPolicy`.*

## References

1. [RL Course by David Silver, Lecture 7: Policy Gradient](https://www.youtube.com/watch?v=KHZVXao4qXs)
2. [LunarLander-v2 environment details](https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2)
