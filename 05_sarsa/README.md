# Module 5: SARSA

The GLIE Monte Carlo method that we discussed in the last module has high variance. In this module, we will explore the 
balance of bias and variance in Reinforcement Learning. We will first discuss an online algorithm called SARSA(0), which 
has low variance (but high bias). Later, we will talk about an algorithm called SARSA(lambda), which strikes a balance 
between GLIE Monte Carlo and SARSA(0).

## Lab

We will code up two agents that solves the CartPole-v0 environment using the SARSA(0) and SARSA(lambda) algorithms. We 
will also explore the effect of learning rate and its annealing on overall performance.

# Required reading

This section is for students who are financially disadvantaged and are not able to pay for the full course. You should still be able to learn a lot by going through the reading material listed below and solving the assignments.

The reading materials are a collection of free online resources that cover many of the things that I address in the course videos (though not everything unfortunately). It's best to go through them in the order listed. 

After you are done studying the listed materials, try solving the assignments.

1. [RL Course by David Silver, Lecture 5: Model Free Control (From 38:46)](https://www.youtube.com/watch?v=lfHX2hHRMVQ&t)
