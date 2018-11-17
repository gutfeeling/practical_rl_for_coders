# Module 5: SARSA

At the end of this module

1. You will be able to explain the weaknesses of the GLIE Monte Carlo algorithm in terms of bias and variance 
2. You will implement a Reinforcement Learning algorithm called SARSA(lambda), which allows you to tune the 
level of bias and variance :tada:
3. You will be able to explain why a constant learning rate is undesirable, and integrate learning rate schedules
in your code

## Plan

1. :movie_camera: Bias vs. variance
2. :movie_camera: The SARSA(0) update
3. :movie_camera: Learning rate and its schedule
4. :pencil: **Exercise**: *Modify `GLIEMonteCarloAgent` to include a function for learning rate schedule*
5. :pencil: **Exercise**: *Modify `GLIEMonteCarloAgent` and  to use the SARSA(0) updates*
6. :movie_camera: SARSA(lambda)
7. :pencil: **Exercise**: *Modify `ModelAndPolicy` to add a table for eligibility traces and implement the
SARSA(lambda) update*
8. :pencil: **Exercise**: *Compare the variance of performance for SARSA and SARSA(lambda).
Which one is better?*

## References

1. [RL Course by David Silver, Lecture 5: Model Free Control (From 38:46)](https://www.youtube.com/watch?v=lfHX2hHRMVQ&t)

## Navigation

[Next - Module 6: Function approximation (Tile Coding)](https://github.com/gutfeeling/practical_rl_for_coders/tree/master/06_fn_approx_tile_coding)

[Previous - Module 4: GLIE Monte Carlo](https://github.com/gutfeeling/practical_rl_for_coders/tree/master/04_glie_monte_carlo)
