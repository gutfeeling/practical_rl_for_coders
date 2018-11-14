#  Practical Reinforcement Learning 

The following is the course plan for my upcoming course on Reinforcement Learning. 

## Course Plan

### Module 1: RL Breakthroughs and Use Cases

1. Welcome to the course!
2. Reinforcement Learning breakthroughs
3. Who is using RL in the industry?

### Module 2: RL basics with `Open AI Gym`

1. What is `OpenAI Gym`?
2. `OpenAI Gym` installation
3. Agent, Environment, State, Action, Rewards
4. Policy and discount rate
5. **Exercise**: *Implement a training and testing loop for a random policy in the CartPole-v0 environment.*
6. The importance of baselines
7. `OpenAI Gym` environment wrappers
8. **Exercise**: *Wrap the `CartPole-v0` environment to change the reward signature. The agent should get -1 reward if the
pole angle exceeds 15 degrees and 0 otherwise.*
9. Monitoring using `OpenAI Gym`
10. **Exercise**: *Implement automatic monitoring for the training and testing loops you wrote earlier.*
11. Supervised learning vs. RL - when to use which?

### Module 3: Bellman Equations

1. Markov processes
2. Value function and Q value function
3. The Bellman equation and optimality theorem
4. Value iteration
5. **Exercise**: *Wrap the `CartPole-v0` environment to make it amenable to value iterations. Discretize observations to 
first place of decimal.*
6. **Exercise**: *Given a policy, find the values for a list of states.*

### Module 4: GLIE Monte Carlo

1. The Monte Carlo Update
2. How to Structure RL Code
3. **Exercise**: *Write an `Agent` class which will collect experiences and pass it on to a `ModelAndPolicy` class.
The `ModelAndPolicy` class should implement a random policy.*
4. **Exercise**: *Modify `ModelAndPolicy` so that it uses a table for modeling the environment. Update the model using Monte
Carlo updates.*
5. Exploration vs. Exploitation - epsilon greedy policies and epsilon annealing
6. **Exercise**: *Modify the `Agent` class to create an epsilon schedule during training.*
7. **Exercise**: *Modify the `ModelAndPolicy` class to implement an epsilon greedy policy.*
8. Creating the hyperparameter file
9. Importance of reproducibility in RL
10. **Exercise**: *Add command line options related to reproducibility in the starter script. Use `argparse`. Run 
the agent twice and check if results are reproducible.*

### Module 5: SARSA

1. Bias vs. variance
2. The SARSA(0) update
3. Learning rate and its schedule
4. **Exercise**: *Modify `GLIEMonteCarloAgent` to include a function for learning rate schedule.*
5. **Exercise**: *Modify ``GLIEMonteCarloAgent` and `ModelAndPolicy` to use the SARSA(0) updates.*
6. SARSA(lambda)
7. **Exercise**: *Modify `ModelAndPolicy` to add a table for eligibility traces and implement the 
SARSA(lambda) update.*
8. **Exercise**: *Compare the performance and its variance over episodes for SARSA and SARSA(lambda). 
Which one is better?*

### Module 6: Function approximation (Tile Coding)

1. **Exercise**: *Try to solve `MountainCar-v0` using GLIE Monte Carlo, SARSA(0) and SARSA(lambda). Explain the agent's
performance on both algorithms.*
2. Limitations of table based methods
3. Tables to functions
4. Linear functions - features, weights, gradient descent and update rule
5. A simple linear function - Tile Coding with Sutton's tile coding package
6. **Exercise**: *Implement a method in `ModelAndPolicy` to extract tile coding features from the
observations of `MountainCar-v0`.*
7. **Exercise**: *Modify `ModelAndPolicy` to include a instance variable for weights and write a method
that returns the Q value by using linear function approximation.*
8. **Exercise**: *Modify the update rule in `ModelAndPolicy`*
9. **Exercise**: *Visualize the learned function using Matplotlib*
10. Theoretical guarantees of convergence

### Module 7: Function approximation (Fourier Series)

1. Fourier Series Refresher
2. **Exercise**: *Find the Fourier coefficients upto order 5 for a square wave.*
3. Using Fourier Series in RL function approximation - features, weights, updates, learning rate
4. **Exercise**: *Wrap the `Acrobot-v1` environment to reduce the observation space from 6 dimensional to 4 dimensional by
converting cartesian co-ordinates to polar co-ordinates.*
5. **Exercise**: *Implement a method in `ModelAndPolicy` to extract fourier coefficients from the
observations of `Acrobot-v0`.*
6. **Exercise**: *Modify `ModelAndPolicy` to include a instance variable for weights, an instance variable for
learning rate per feature and write a method that returns the Q value by using linear function approximation.*
7. **Exercise**: *Modify the update rule in `ModelAndPolicy`.*

### Module 8: Neural Network crash course with `Keras`

1. Multi layer perceptrons
2. What is `Keras`?
3. `Keras` installation
4. Creating an MLP using `Keras` - Model and Functional API
5. Activations - Linear, Sigmoid and ReLU
6. Activations in `Keras`
7. Loss
8. Losses in `Keras`
9. Implementing custom loss functions
10. Backpropagation
11. Optimizers in `Keras` - RMSProp and Adam
12. Fitting in `Keras` - epoch, batch, minibatch
13. Monitoring in `Keras` using callbacks
12. **Exercise**: *Implement an MLP in `Keras` that learns to return the sum of squares of an input array.*
13. Convolutional neural networks

### Module 9: Function approximation (Neural Network)

1. Neural Networks as function approximators
2. **Exercise**: *Modify `ModelAndPolicy for `CartPole-v0` to use a `Keras` model for function approximation.*
3. **Exercise**: *Modify the update rule in `ModelAndPolicy` to implement `Keras model` fitting based on
SARSA(0).*
4. Theoretical limits on convergence for Neural Networks as function approximators

### Module 10: Vanilla Policy Gradient

1. Limitations of using value based methods - continuous action spaces
2. Policy iterations and the score function
3. Discrete actions - Softmax policy
4. Continuous action spaces - Gaussian policy
4. **Exercise**: *In the `LunarLander-v2` environment, modify `ModelAndPolicy` to include a function that 
gets the features.*
4. **Exercise**: *Replace epsilon greedy action with a softmax policy.*
5. REINFORCE and its limitations
6. Actor Critic architectures
7. **Exercise**: *Implement a critic in `ModelAndPolicy` using a neural network.*
8. **Exercise**: *Implement the actor critic update rule in `ModelAndPolicy`.*

### Module 11: Proximal Policy Approximation

1. Stability of learning
2. Sample efficiency
3. Motivation behind PPO updates and comparison to other methods
4. PPO Loss
5. **Exercise**: *Implement a Neural Network with the PPO loss in `ModelAndPolicy`.*
6. Generalized Advantage Estimation (GAE)
7. **Exercise**: *Modify the `Agent` class to return GAE experiences.*
8. **Exercise**: *Implement the GAE update rule in `ModelAndPolicy`.*


### Module 12: RL On `Google Cloud`

1. Neural Networks : CPU or GPU?
2. Setting up a RL environment on `Google Cloud` with GPUs
3. **Exercise**: *Run the PPO code on your computer's CPU and on a GPU instance on `Google Cloud` and compare speed of
execution in non reproducible mode and reproducible mode.*

### Module 13: Deep Q Network

1. `Atari` environments on `OpenAI Gym`
2. DQN NIPS Preprocessing
3. **Exercise**: *Wrap the `Atari` envs to implement the DQN NIPS preprocessing.*
4. CNN architecture
5. **Exercise**: *In `ModelAndPolicy`, use the CNN architecture used by DQN NIPS.*
4. Off policy learning
5. Q Learning update
6. Replay memory
7. **Exercise**: *Implement a replay memory and make the `Agent` pass the replay memory to `ModelAndPolicy`.*
8. **Exercise**: *Implement Q learning updates on random minibatches in the replay memory in `ModelAndPolicy`.*
7. Tail chasing and target networks- DQN Nature
