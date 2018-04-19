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

2. Provide a script that trains the agent and accepts **all** implementation parameters, including hyperparameters and seeds,
as command line arguments. 

```
python run_ppo_agent --discount_factor 0.99 --lambda_value 0.95 ...
```

3. Provide a saved agent with good performance. 
4. Provide a script that tests the saved agent.

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

### Easy to read and understand code

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

### Extending the implementation











