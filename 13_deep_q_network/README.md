# Module 13: Deep Q Network

At the end of this module

1. You will able to list the main achievements in the Atari DQN papers (both NIPS and Nature versions) by DeepMind
2. You will be able to select the correct Atari environments in `OpenAI Gym` to reproduce Deepmind results
3. You will write environment wrappers in `OpenAI Gym` to reproduce the preprocessing in the NIPS paper
4. You will write a `Keras` model to reproduce the CNN used by the NIPS paper
5. You will be able to explain why replay memory and target networks lead to more stable learning when using Deep Networks
6. You will implement the breakthough DQN algoithm (NIPS version) to solve the `PongNoFrameSkip-v4` environment :tada:

## Plan 

1. :movie_camera: `Atari` environments on `OpenAI Gym`
2. :movie_camera: DQN NIPS Preprocessing
3. :pencil: **Exercise**: *Wrap the `Atari` envs to implement the DQN NIPS preprocessing.*
4. :movie_camera: CNN architecture
5. :pencil: **Exercise**: *In `ModelAndPolicy`, use the CNN architecture used by DQN NIPS.*
4. :movie_camera: Off policy learning
5. :movie_camera: Q Learning update
6. :movie_camera: Replay memory
7. :pencil: **Exercise**: *Implement a replay memory and make the `Agent` pass the replay memory to `ModelAndPolicy`.*
8. :pencil: **Exercise**: *Implement Q learning updates on random minibatches in the replay memory in `ModelAndPolicy`.*
7. :movie_camera: Tail chasing and target networks- DQN Nature

## References

1. [Atari environment variants in OpenAI Gym (Notice how there are many different variants of each environment. For 
reproducing Deepmind results, you would need `NoFrameSkip-v4`)](https://github.com/openai/gym/blob/master/gym/envs/__init__.py)
2. [Meaning of actions in the Atari envs (look at the `ACTION_MEANING` variable)](https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py)
3. [Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)
4. [DQN Preprocessing using `OpenAI Gym` wrappers](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)
5. [DQN implementation using Python and Theano (particularly good as a hyperparameter reference for NIPS and Nature versions)](https://github.com/spragunr/deep_q_rl)
6. [Deep RL Bootcamp, Lecture 3: DQN and variants by Vlad Mnih](https://www.youtube.com/watch?v=fevMOp5TDQs)
7. [DQN NIPS Paper](https://arxiv.org/abs/1312.5602)
8. [DQN Nature Paper](https://www.nature.com/articles/nature14236)


