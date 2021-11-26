# DRL-ContinuousControl
Udacity continuous control reinforcement learning

## Project's goal

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

![In Project 2, train an agent to maintain its position at the target location for as many time steps as possible.](./reacher.gif)

### Environment details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). Unity ML-Agents is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.


The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

- Set-up: Double-jointed arm that will move to various locations given actions.
- Goal: The agents must move the end of the arm to a goal location (i.e. a hand touching a moving ball).
- Agents: The provided Udacity enviroment allows a Single Agent in a single environment or 20 Agents in 20 environments.
- Agent Reward Function (independent):
  - +0.1 Each step agent's hand is in goal location.
- Observation State:
  - 33 variables corresponding to position, rotation, velocity, and angular velocities of the two-joint arm
- Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
- Desired Reward: A good agent should achieve a total reward of ~30 over the course of an episode.

For this project, Udacity provides two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Solving the environment

Depending on the chosen environment, there are 2 options to solve the environment:

**Option 1: Solve the First Version**

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes. 

**Option 2: Solve the Second Version**

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically:
    - After each episode, the rewards that each agent received (without discounting) are added up , to get a score for each agent. This yields 20 (potentially different) scores. The average of these 20 scores is then used.
    - This yields an average score for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

**For this project we used Option 2 of the environment (20 Agents) using DDPG algorithm.** 

## Getting started

### Installation requirements

- For this project, I used the workspace provided by Udacity, so there were no installation requirements.

## Instructions

### Training an agent
    
Run `Continuous_Control.ipynb` in the Udacity Online Workspace.  This will train the agent.  The notebook completed execution in 5-6 hours.   
