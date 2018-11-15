# Module 4: Function Approximation (Tile Coding)

At the end of this module

1. You will be able to decide when and when not to use tabular methods for environment modeling
2. You will be able to explain why function approximation is needed for environments with large state spaces or sparse
rewards
3. You will implement linear function approximation and tile coding to solve the `MountainCar-v0` environment :tada: 
4. You will be able to give examples of situations where function approximation guarantees optimal behavior and 
   situations where there are no guarantees (and the only way out is trial-and-error) 

## Plan

1. :pencil: **Exercise**: *Try to solve `MountainCar-v0` using GLIE Monte Carlo, SARSA(0) and SARSA(lambda). Explain the agent's
performance on these three algorithms.*
2. :movie_camera: Limitations of table based methods
3. :movie_camera: Tables to functions
4. :movie_camera: Linear functions - features, weights, gradient descent and update rule
5. :movie_camera: A simple linear function - Tile Coding with Sutton's tile coding package
6. :pencil: **Exercise**: *Implement a method in `ModelAndPolicy` to extract tile coding features from the
observations of `MountainCar-v0`.*
7. :pencil: **Exercise**: *Modify `ModelAndPolicy` to include a instance variable for weights and write a method
that returns the Q value by using linear function approximation.*
8. :pencil: **Exercise**: *Modify the update rule in `ModelAndPolicy`*
9. :pencil: **Exercise**: *Visualize the learned function using `matplotlib`*
10. :movie_camera: Theoretical guarantees of convergence

## References

1. [Reinforcement Learning: An Introduction, Second Edition, Section 9.1 - 9.4 & Section 9.5.4](https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view)
2. [Tile Coding Software - Reference Manual, Version 2.1, Python version](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html)
