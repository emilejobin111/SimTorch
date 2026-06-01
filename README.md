# SimTorch
SimTorch is a Machine learning library based on numpy and jax with a focus on reinforcement learning. It has multiple sub module for diferent use case.

# SimTorch Documentation

# Intro
`SimTorch` is a Machine learning library based on numpy with a focus on reinforcement learning, 

## 1. `SimTorch.core`
The `core` module contain the basics of `SimTorch` such as the neural networks, gradient optimizers and schedulers.

## 2. `SimTorch.rl`
The `rl` module is the deep reinforcement learning module of `SimTorch` it uses the core module to build and train neural networks on some environnement, the envs are defined using the `gymnasium` api. The `rl` module provides three different type of optimisation method :
- Soft actor critic (off-policy algorithm)
- Proximal Policy Optimization (on-policy algorithm)
- Open AI evolution strategies (2017 evolution strategie algorithm developed by Open AI as an alternative to reinforcement learning)

## 3. `SimTorch.save`
The `save` module is made to save/load training optimisation method.