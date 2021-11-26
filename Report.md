# DRL-ContinousControl

[//]: # (Image References)

![In Project 2, train an agent to maintain its position at the target location for as many time steps as possible.](./reacher.gif)


# Project 2: Continuous Control

### Algorithm

For this project I implemented the Deep Q-Network as described in the paper: [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

This approach moves beyond standard online Q-Learning in two respects:
* Experience replay
* Use a separate network for generating the targets in the Q-learning update

Both of these improvements help to address instabilities associated with Q-Learning.  The paper describes it as follows:

> We address these instabilities with a novel variant of Q-learning, which
> uses two key ideas. First, we used a biologically inspired mechanism
> termed experience replay that randomizes over the data, thereby
> removing correlations in the observation sequence and smoothing over
> changes in the data distribution. Second, we used
> an iterative update that adjusts the action-values (Q) towards target
> values that are only periodically updated, thereby reducing correlations
> with the target.

The pseudo-code from the paper is as follows:

![alg](screencapture.png)

### Implementation

The code used for this project is heavily based on the solution in the Deep Q-Learning Lunar Lander project.  The key components are as follows:

#### model.py:  
This file contains the QNetwork class, which implements a pytorch deep neural network that takes input the size of the state space, produces output the size of the action space, and has two hidden layers (with ReLU activation) of configurable size.  For this project, both layers were set to 512 nodes.  As described in the algorithm section, the QNetwork(s) were used for both training and inference at various points.

#### dqn_agent.py:
This file contains two classes.  The Agent class is the main interface for interacting with and learning from the enviroment.  The Agent delegates to the ReplayBuffer class to store the experience tuples needed for experience replay.  The key methods are described below.
##### Agent.__init__
A key feature of this constructor is that it instantiates two QNetworks, qnetwork_local and qnetwork_target.  While qnetwork_local is being trained, qnetwork_target is used to generate stable targets for computing the loss.
##### Agent.act
This method returns an action based on the input state.  It makes an epsilon greedy selection based on the prediction from qnetwork_local.  Since epsilon is an input parameter, the training loop can gradually move from exploration to exploitation.
##### Agent.step
The training loop will choose an action and provide it to the enviroment.  With the environment's response, we have a full experience (state, action, reward, next_state, done) that can be stored in the ReplayBuffer.  For every UPDATE_EVERY calls to Agent.step, the method will sample BATCH_SIZE samples from the ReplayBuffer and update both QNetworks by calling Agent.learn
##### Agent.learn
This method implements the target computation and gradient descent step from the algorithm pseudo-code, with the modification of using an Adam optimizer.  The updated qnetwork_local is then used to gradually update (based on interpolation parameter TAU) the qnetwork_target.
#### Network.ipynb
The dqn() method in the notebook is the main training loop.  It uses Agent.act to choose actions, uses the actions to generate experiences from the enviroment, and feeds the experiences to Agent.step, where the learning is triggered after UPDATE_EVERY steps.
### Hyperparameters
The following hyperparameter settings were used:
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```
The model architecture for the neural network is described above in the model.py section.

### Plot of Rewards
[Navigation.ipynb](https://github.com/stevenapsel/DRL-Navigation/blob/main/Navigation.ipynb) shows the plot of rewards and the number of episodes required.  As shown below, it took 707 episodes to achieve an average score of 13.
```
Episode 100	Average Score: 1.12
Episode 200	Average Score: 4.19
Episode 300	Average Score: 6.30
Episode 400	Average Score: 8.56
Episode 500	Average Score: 10.32
Episode 600	Average Score: 11.99
Episode 700	Average Score: 12.51
Episode 800	Average Score: 12.63
Episode 807	Average Score: 13.00
Environment solved in 707 episodes!	Average Score: 13.00
```

### Ideas for Future Work
Here are a few ideas that could improve the training speed and/or performance of the agent.
#### Hyperparameter Tuning
The hyperparameters we've used here were fine for completing the project.  Are they optimal?  Probably not.  Some additional exploration could yield better results.
#### Network Architecture
We used a fairly simple network in QNetwork.  Further exploration could look at varying the size (or number) of the hidden layers.
#### Prioritized Experience Replay
For this project, our replays were uniformly sampled.  What if we could choose sampled from a weighted distribution that gave preference to experiences that are more likely to have an larger impact on learning?  That is the idea behind [prioritized experience replay](https://arxiv.org/abs/1511.05952).
