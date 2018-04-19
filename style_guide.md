# RL for Coders Style Guide

This document describes guidelines for implementing RL algorithms. These rules 
follow directly from the goals stated below.

## Goals

1. An open source implementation gets value when people use it. => *Should be easy to install and run.*

2. Reproducibility is a big issue in RL. => *Results should be perfectly reproducible.*

3. Guido said code is more often read than written. => *Code should be easy to read and understand.*

3. Deep RL is a relatively new field, and there's a lot of experimentation yet to be done. => *Should be easy to extend.*

4. Every RL practitioner has faced the situation where the agent simply refuses to learn, even when it is known that it should.
It's usually a bug and looking for discrepancies in the logs is a great way to find it. => *Implementation should generate 
plentiful logs.*

5. Images/videos are usually easier to comprehend than  text => *The reproducible benchmark should be accompanied by plots and 
videos of performance.*

## Non-goals

It is impossible to optimize for everything. So here is a non goal.

1. We are not optimizing for speed. Speed optimization involves low level packages and distributed agents which make the 
implementation more complex to read and understand. In general, we stick to the principle "Premature optimization is the root 
of all evil".

## Guidelines

So how do we ensure that we meet our goals? Here are some guidelines. 

### Installing and running the code
 
1. Installing should be as simple as 

```
pip install -e ppo
```

2. In case installation involves additional steps (e.g. installing system packages), document the steps clearly.

3. Provide a script that trains the agent and accepts **all** implementation parameters, including hyperparameters and seeds,
as command line arguments. 

```
python run_ppo_agent --discount_factor 0.99 --lambda_value 0.95 ...
```

4. Provide a saved agent with good performance. 

5. Provide a script that tests the saved agent.

```
python test_saved_ppo_agent --saved_actor_model_file_path /path/to/actor/model
```

### Reproducibility

1. Accept seeds (for initializing **any and all** random number generation in the implementation) as command line arguments. 

```
python run_ppo_agent --make_reproducible True --numpy_seed 42 --random_seed 1000 --tensorflow_seed 27 ...
```

2. Save the list of all parameter values used during training to a log file.

```
- training_logs
    - parameters.json # all parameters for this training run, including hyperparameters and seeds should be logged here
    - other logs go here ...
```
3. Include logs from a reproducible run along with the implementation.

### Easy to read and understand

1. Follow [PEP8](https://www.python.org/dev/peps/pep-0008/). 

2. Document all classes and functions using docstrings.

3. Use comments and block quotes generously to explain code.

4. Avoid unnecessary verbosity. Prefer high level packages like Keras over Tensorflow and Gym over ALE whenever possible.

5. Err on the side of flat folder structure when organizing files.

6. Use expressive names for variables, functions, and classes. The name should communicate what they are or what they do.

```python
# variable example
total_rewards_obtained_in_this_episode = 0

# function example
def compute_advantages_and_value_targets(...):
    # implementation goes here
    
# class example 
class DefaultActor(object):
    # implementation goes here
```

7. Follow conventions in RL literature when naming variables. If it is a discount factor, then call the variable 
`discount_factor` or `gamma`. 

8. There should be one (and only one) source of truth for hyperparameters. This should preferably be the script for training 
the agent. No other file should set or override hyperparameters or provide defaults.

### Extendibility

1. Agent should be as environment and model agnostic as possible. In practice, agents should assume a standard interface for function approximation models and environments and should not make any more assumptions. For example, agents can assume that the function approximation model will have a `get_action(observations)` but should not make any assumptions about the dimension of the returned action. Similarly, agents can assume `env.step(action)` will return `observation, reward, done, info` but should 
not make any assumptions about the dimension or range of observations. If absolutely required, use methods like  `env.action_space.low` or `env.observation_space.shape` etc instead of hardcoding environment specific details.

2. Function approximation models (e.g. Neural nets) should be a pluggable component of the agent and should define a standard 
interface that the agent can assume. Use a separate file for keeping models, preferably called `models.py`.

3. Any modification to the vanilla Gym environment (such as grayscaling images, scaling rewards etc.) should be done using a 
Gym Wrapper. The wrapped environment should be passed to the agent. The agent should never make any modifications to `observations`, `rewards` or `done`. The wrapped environments should be defined in a separate file, preferably called 
`env_wrappers.py` and should expose the same interface as the vanilla Gym environments.

### Logging

1. Use automatic logging provided by high level packages like Keras and Gym instead of writing your own code whenever possible.

```
from keras.callbacks import ModelCheckpoint

model.fit(inputs, targets, callbacks = [ModelCheckpoint(),])
```

```
from gym.wrappers import Monitor

env = gym.make("BipedalWalker-v2")
monitored_env = Monitor(env, "/path/to/gym/log/directory")
```

2. Generate as much logs as possible.

3. Print well formatted and informative updates to the command line when the agent trains.

### Presentation

1. Generate informative plots from the logs. Generate videos of the agent solving the environment. Use plots and videos to present results.









