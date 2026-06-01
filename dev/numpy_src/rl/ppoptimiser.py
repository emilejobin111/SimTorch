"""Proximal Policy Optimization implementation for SimTorch.

This module contains the rollout buffer, PPO training loop, hyperparameter containers,
checkpoint helpers, and policy/value update routines.
"""

from __future__ import annotations
from typing import override, Tuple, Any, Callable, Optional, TypedDict
from dataclasses import dataclass
import os
from abc import ABC, abstractmethod
from copy import deepcopy
import time

import numba
from numba import njit
import numpy as np
import gymnasium as gym

from ..core import nn
from ..core import optim
from . import policy
from ..core import scheduler
from ..core.rng import rng

from ..save.save_nn import CHECKPOINT_DIR, write_metadata, save_optimizer_states, read_metadata, load_optimizer_states
from pathlib import Path

from .base_nn_optimiser import NetworkOptimiser
from .utils import (Normalizer, 
                    NoNormalizer, 
                    RunningMeanStd, 
                    LOG_2_PI,
                    normalize_values,
                    compute_log_prob,
                    print_red_warning_simple)








class BufferException(Exception):
    """Exception raised when the buffer workflow encounters an invalid state.
    
    Example:
        >>> obj = BufferException()
    """

    ...


class RolloutBuffer:
    """Memory buffer to store environment transitions and agent interactions for PPO.
    
    This buffer handles vectorized environments (a population of agents) and stores
    observations, actions, rewards, values, log probabilities, and done flags over a fixed
    number of steps (rollout_size). It also provides highly optimized static methods using
    Numba to compute log probabilities and Generalized Advantage Estimation (GAE).
    
    Args:
        obs_dim_count (int): The number of features in a single observation.
        action_dim_count (int): The number of dimensions in a single action.
        rollout_size (int): The number of steps to collect per environment before training.
        population_size (int): The number of parallel environments/agents.
    
    Example:
        >>> obj = RolloutBuffer(obs_dim_count=4, action_dim_count=4, rollout_size=4, ...)
    """


    def __init__(self, obs_dim_count:int, action_dim_count:int, rollout_size:int, population_size:int):
        """Initialize a new RolloutBuffer instance.
        
        Args:
            obs_dim_count (int): Number of scalar features in one observation.
            action_dim_count (int): Number of scalar features in one action.
            rollout_size (int): Maximum number of transitions stored by the buffer.
            population_size (int): Number of parallel environments represented in the rollout.
        """
        self.__rollout_size = rollout_size
        self.__index = 0
        self.__observations = np.zeros((rollout_size, population_size, obs_dim_count), dtype=np.float32)
        self.__actions = np.zeros((rollout_size, population_size, action_dim_count), dtype=np.float32)
        self.__values = np.zeros((rollout_size, population_size), dtype=np.float32)
        self.__rewards = np.zeros((rollout_size, population_size), dtype=np.float32)
        self.__log_probs = np.zeros((rollout_size, population_size), dtype=np.float32)
        self.__dones = np.zeros((rollout_size, population_size), dtype=np.bool_)
    
    @property
    def is_full(self) -> bool:
        """Gets if the rollout buffer is full.
        
        Returns:
            bool: if the rollout buffer is full.
        """
        return self.index == self.rollout_size
    
    @property
    def rollout_size(self) -> int:
        """Gets the maximum capacity of the buffer per environment.
        
        Returns:
            int: The total number of steps the buffer can hold.
        """
        return self.__rollout_size
        
    @property
    def index(self) -> int:
        """Gets the current step index in the buffer.
        
        Returns:
            int: The current insertion index.
        """
        return self.__index
        
    @property
    def observations(self) -> np.ndarray:
        """Gets the stored observations.
        
        Returns:
            np.ndarray: Array of shape (rollout_size, population_size, obs_dim_count).
        """
        return self.__observations
        
    @property
    def actions(self) -> np.ndarray:
        """Gets the stored actions.
        
        Returns:
            np.ndarray: Array of shape (rollout_size, population_size, action_dim_count).
        """
        return self.__actions
        
    @property
    def values(self) -> np.ndarray:
        """Gets the stored state values estimated by the Critic.
        
        Returns:
            np.ndarray: Array of shape (rollout_size, population_size).
        """
        return self.__values
        
    @property
    def rewards(self) -> np.ndarray:
        """Gets the stored environment rewards.
        
        Returns:
            np.ndarray: Array of shape (rollout_size, population_size).
        """
        return self.__rewards
        
    @property
    def log_probs(self) -> np.ndarray:
        """Gets the stored action log probabilities.
        
        Returns:
            np.ndarray: Array of shape (rollout_size, population_size).
        """
        return self.__log_probs
        
    @property
    def dones(self) -> np.ndarray:
        """Gets the stored terminal state flags.
        
        Returns:
            np.ndarray: Array of shape (rollout_size, population_size).
        """
        return self.__dones
    
    def fill_values_buffers(self, critic: policy.CriticNN):
        """Evaluates all stored observations using the Critic network and fills the values buffer.
        
        The method operates on preallocated storage so experience collection can proceed without
        repeated memory allocation.
        
        Args:
            critic (policy.CriticNN): The Critic neural network used to estimate state values.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.fill_values_buffers(critic=...)
        """
        flat_obs = self.__observations.reshape(-1, self.__observations.shape[-1])
        flat_values = critic.forward(flat_obs)
        self.__values[:] = flat_values.reshape(self.__rollout_size, -1)
    
    
    def reset_buffer(self):
        """Resets the internal pointer to the beginning of the buffer.
        
        This prepares the buffer for a new rollout collection phase without needing to
        reallocate memory.
        
        Returns:
            None: This method resets the buffer state in place and returns no value.
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.reset_buffer()
        """
        self.__index = 0
    
    def add_step(self,
                 observations:np.ndarray,
                 actions: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 means: np.ndarray,
                 stds: np.ndarray):
        """Records a single environment transition for the entire population.
        
        Automatically computes and stores the log probability of the taken actions based on the
        provided means and standard deviations.
        
        Args:
            observations (np.ndarray): The current states of shape (population_size,.
                                       obs_dim_count).
            actions (np.ndarray): The sampled actions of shape (population_size,.
                                  action_dim_count).
            reward (np.ndarray): The rewards received of shape (population_size,).
            done (np.ndarray): The terminal flags of shape (population_size,).
            means (np.ndarray): The mean action values predicted by the Actor.
            stds (np.ndarray): The standard deviations predicted by the Actor.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Raises:
            BufferException: Raised if the rollout buffer is accessed in an invalid state.
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.add_step(observations=..., actions=..., reward=..., ...)
        """
        
        if self.__index >= self.__rollout_size:
            raise BufferException("""the rollout buffer is full and `add_step` has been called, can't add step to a full buffer. """)
        
        self.__observations[self.__index] = observations
        self.__actions[self.__index] = actions
        self.__rewards[self.__index] = reward
        self.__dones[self.__index] = done
        #^ we just need the log prob and this simple value is way smaller than two action and obs vector so we calculate it here
        self.__log_probs[self.__index] = compute_log_prob(means,stds,actions)
        
        self.__index += 1
    def get_mean_score(self):
        """Return mean score.
        
        The method operates on preallocated storage so experience collection can proceed without
        repeated memory allocation.
        
        Returns:
            Any: Average completed-episode score represented by the rollout.
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.get_mean_score()
        """
        mean = self.__get_mean_score(rewards=self.__rewards,
                                     dones=self.__dones)
        if np.isnan(mean):
            return None
        return float(mean)
    @staticmethod
    @njit(cache=True,) #parallel=True) <--- for a small amount of data, parallel is not worth it
    def __get_mean_score(rewards: np.ndarray, dones: np.ndarray):
        """Internal helper that handles get mean score.
        
        Args:
            rewards (np.ndarray): Reward values associated with the current transitions.
            dones (np.ndarray): Boolean flags indicating episode termination.
        
        Returns:
            Any: Computed result returned by the function.
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.__get_mean_score(rewards=..., dones=...)
        """
        by_pop_mean = np.empty(shape=(rewards.shape[1],),dtype=np.float32)
        for j in numba.prange(rewards.shape[1]):
            tot = 0.0
            run_tot = 0.0
            n = 0
            for i in range(rewards.shape[0]):
                run_tot += rewards[i,j]
                if dones[i,j]:
                    if n != 0:
                        tot += run_tot
                    run_tot = 0.0
                    n += 1
            if n <= 1:
                by_pop_mean[j] = np.nan
            else:
                by_pop_mean[j] = tot/(n-1)
        mean = np.nanmean(by_pop_mean)
        return mean
    
    @staticmethod
    @njit(cache=True,)
    def __compute_GAE(rewards: np.ndarray,
                      values: np.ndarray,
                      dones: np.ndarray,
                      last_value: np.ndarray,
                      gamma: float = 0.99,
                      lambd: float = 0.9 #! cant use `lambda` because of python :(
                      ) -> Tuple[np.ndarray,np.ndarray] :
        """Computes Generalized Advantage Estimation (GAE) and target critic values.
        
        Calculates the advantages for policy updates and the target returns (td-targets) for
        Critic training, using backward iteration. Compiled with Numba for speed.
        
        Args:
            rewards (np.ndarray): The collected rewards over the rollout.
            values (np.ndarray): The state values estimated by the Critic.
            dones (np.ndarray): Flags indicating episode termination.
            last_value (np.ndarray): The Critic's value estimate for the final state after the.
                                     rollout.
            gamma (float): The discount factor. Defaults to 0.99.
            lambd (float): The GAE smoothing parameter (lambda) for bias-variance tradeoff.
                           Defaults to 0.9.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing: - The computed advantages (GAE).
                                           - The target values for the Critic to learn from
                                           (critic_goal).
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.__compute_GAE(rewards=..., values=..., dones=..., ...)
        """
        
        #^ --- variables initialisation ---
        #? we need to know if it's still playing and not if it ended
        playing = 1-dones
        gae = np.zeros_like(rewards)
        sigma = np.zeros_like(rewards)
        #? We need the last "guess" of the Critic because we dont have it from the rollout phase
        values_and_last:np.ndarray = np.vstack((values,last_value.reshape(1,-1)))
        #^ --------------------------------
        
        #^ --- the δ (sigma) calulation ---
        #? application of this maths formula:
        #? δ_t​ = rt ​+ γ * v_{t+1} ​* (1−d_t​) − v_t​ or:
        #? sigma_t = rewards_t + gamma * (the guessed expected value at t + 1) - (the guessed expected value at t)
        sigma[:] = rewards + gamma * values_and_last[1:] * playing - values_and_last[:-1]
        #& it's pretty much "how good the action was compared to the expected value"
        #^ --------------------------------
        
        #^ --- Bootstraping ---
        #& the gae at t is just how good was the action at t
        gae[-1] = sigma[-1]
        #^ --------------------
        
        #^ --- The main gae computation ---
        #& we want to know how good was an action
        #& taking into consideration the future rewards the Actor gets
        #? so we start from the last step because we need to know if the actor did good after the action and not before
        for t in range(values.shape[0] - 2, -1, -1):
            #? the gamma * lambda does that the Actor is more rewarded for an action he just did (+10 now is better than +10 later)
            gae[t] = sigma[t] + (gamma * lambd) * playing[t] * gae[t+1]
            #? the * playing[t] makes it so that if the episode ended, we do * 0
            #? So we take into consideration just the episode he was playing on.
            #? and not all the ones after wich have no connection to the first one
        #^ --------------------------------
        
        #^ --- the target return of the Critic ---
        #? the gae is also how far the Critic was from guessing the right value
        #? If the critic was perfect, the gae would be 0
        #? So to train it, we need to know what is the right anwer, wich is gae + values here
        critic_goal = gae + values_and_last[:-1]
        #^ ----------------------------------------
        return gae, critic_goal
    
    
    def compute_GAE(self,
                    critic: policy.CriticNN,
                    last_obs: np.ndarray, 
                    hyper_params: PPOptimiser.HyperParams
                    ) -> Tuple[np.ndarray,np.ndarray] :
        """Computes the Generalized Advantage Estimation (GAE) for the entire rollout.
        
        This public method evaluates the final state of the environment using the provided
        Critic network to bootstrap the advantage calculation. It then delegates the heavy
        mathematical processing to the JIT-compiled private method.
        
        Args:
            critic (policy.CriticNN): The Critic neural network used to evaluate the. value of.
                                      the final observation.
            last_obs (np.ndarray): The state observation immediately following the. last step.
                                   of. the rollout. Used for bootstrapping.
            hyper_params (PPOptimiser.HyperParams): Hyperparameter object controlling the.
                                                    training loop.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The computed advantages (GAE) of shape (rollout_size,
                                           population_size). The target returns for the Critic
                                           (critic_goal) of shape (rollout_size,
                                           population_size).
        
        Example:
            >>> obj = RolloutBuffer(...)
            >>> result = obj.compute_GAE(critic=..., last_obs=..., hyper_params=...)
        """
        
        last_values = critic.forward(last_obs)
        gae, critic_goal = self.__compute_GAE(
            rewards=self.__rewards,
            values=self.__values,
            dones=self.__dones,
            last_value=last_values,
            gamma=hyper_params.gamma,
            lambd=hyper_params.lambd
        )
        return gae, critic_goal



class PPOptimiser(NetworkOptimiser):
    """Optimizer implementing the pp training procedure.
    
    The implementation coordinates runtime state, optimization steps, and
    checkpoint-friendly serialization helpers needed by the training workflow.
    
    Args:
        env (gym.vector.VectorEnv): Gymnasium environment used for interaction or.
                                    validation.
        networks (nn.Sequential | Tuple[policy.ActorNN, policy.CriticNN]): Collection of.
                                                                           neural networks.
                                                                           used by the.
                                                                           optimizer.
        actor_optimizer (optim.GradientOptimizer): Optimizer used to update the actor.
                                                   network.
        critic_optimizer (optim.GradientOptimizer): Optimizer used to update the critic.
                                                    network.
    
    Example:
        >>> obj = PPOptimiser(env=env, networks=..., actor_optimizer=..., ...)
    """


    def __init__(self, 
                 env: gym.vector.VectorEnv,
                 networks:nn.Sequential|Tuple[policy.ActorNN,policy.CriticNN],
                 actor_optimizer:optim.GradientOptimizer,
                 critic_optimizer:optim.GradientOptimizer,):
        #^ --- type checking ---
        #& gets the actor and critic by the two possible ways
        """Initialize a new PPOptimiser instance.
        
        Args:
            env (gym.vector.VectorEnv): Gymnasium environment used for interaction or.
                                        validation.
            networks (nn.Sequential | Tuple[policy.ActorNN, policy.CriticNN]): Collection of.
                                                                               neural networks.
                                                                               used by the.
                                                                               optimizer.
            actor_optimizer (optim.GradientOptimizer): Optimizer used to update the actor.
                                                       network.
            critic_optimizer (optim.GradientOptimizer): Optimizer used to update the critic.
                                                        network.
        
        Raises:
            SimTorch.core.nn.InitializationException: Raised if the component is used before initialization is.
                                     complete.
            NotImplementedError: Raised if the requested configuration is not implemented.
        """
        if isinstance(networks,nn.Sequential):
            self.__actor, self.__critic = self.get_actor_critic_from_sequential(networks)
        elif isinstance(networks,(list,tuple)):
            self.__actor, self.__critic = self.get_actor_critic_from_list(networks)
        else :
            raise nn.SimTorch.core.nn.InitializationException("networks needs to be a list/tuple of one ActorNN and one CriticNN or a Sequential nn")
        
        if not isinstance(env,gym.vector.VectorEnv):
            raise NotImplementedError("PPO only works with vector env")
        
        if not isinstance(env.single_action_space, gym.spaces.Box):
            raise NotImplementedError("PPO only works with continuous actions spaces")
        else :
            self.__low_action, self.__high_action = env.single_action_space.low, env.single_action_space.high
        #^ ---------------------
        self.__f64_obs = env.single_observation_space.dtype == np.float64
        self.__f64_act = env.single_action_space.dtype == np.float64
        
        
        self.__env = env
        self.__actor_optimizer = actor_optimizer
        self.__critic_optimizer = critic_optimizer
        self.__population_size = env.num_envs
        
        self.__training_ready = False
    
    def get_actor_critic_from_sequential(self,neural_network:nn.Sequential):
        """Return actor critic from sequential.
        
        Args:
            neural_network (nn.Sequential): Network instance used for neural.
        
        Returns:
            Any: Computed result returned by the function.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.get_actor_critic_from_sequential(neural_network=...)
        """
        actor = policy.ActorNN.from_sequential(neural_network)
        critic = policy.CriticNN.from_sequential(neural_network)
        return actor, critic
    
    def get_actor_critic_from_list(self, networks:Tuple[policy.ActorNN,policy.CriticNN]):
        """Return actor critic from list.
        
        Args:
            networks (Tuple[policy.ActorNN, policy.CriticNN]): Collection of neural networks.
                                                               used by the optimizer.
        
        Returns:
            Any: Computed result returned by the function.
        
        Raises:
            SimTorch.core.nn.InitializationException: Raised if the component is used before initialization is.
                                     complete.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.get_actor_critic_from_list(networks=...)
        """
        type_list = [type(network) for network in networks]
        
        if type_list == [policy.ActorNN,policy.CriticNN]:
            return networks
        elif type_list == [policy.CriticNN,policy.ActorNN]:
            return networks[::-1]
        else :
            raise nn.SimTorch.core.nn.InitializationException("networks needs to be a list/tuple of one ActorNN and one CriticNN or a Sequential nn")
        
    def __get_trajectories(self, 
                           rollout_buffer:RolloutBuffer,
                           obs_normalizer: Normalizer,
                           hyper_params: HyperParams) -> Tuple[np.ndarray,np.ndarray]:
        
        """Internal helper that handles get trajectories.
        
        Args:
            rollout_buffer (RolloutBuffer): Buffer instance that stores intermediate rollout or.
                                            replay data.
            obs_normalizer (Normalizer): Observation normalizer applied before network.
                                         evaluation.
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Computed result expressed as `Tuple[np.ndarray,
                                           np.ndarray]`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.__get_trajectories(rollout_buffer=..., obs_normalizer=..., hyper_params=...)
        """
        #~ --- reset the buffer to get new data ---
        rollout_buffer.reset_buffer()
        #~ ----------------------------------------
        
        #~ --- gets the curent states ---
        obs = self.__curent_obs
        info = self.__curent_info
        #~ ------------------------------
        
        while not rollout_buffer.is_full:
            # --- type casting ---
            if self.__f64_obs:
                obs = obs.astype(np.float32)
            # --------------------
            
            #^ --- normalizes obs ---
            norm_obs = obs_normalizer.normalize_and_update(obs)
            #^ ----------------------
            
            #^ --- gets the action normal distr for the action ---
            means, stds = self.__actor.stochastic_forward(norm_obs)
            #^ ---------------------------------------------------
            
            #^ --- randomly chose the next actions ---
            raw_actions = means + stds * rng.normal(0,1,size=stds.shape)
            #^ --------------------------------------
            
            #~ --- limits the actions to the env range ---
            actions = np.clip(raw_actions,self.__low_action,self.__high_action)
            #~ -------------------------------------------
            
            
            # --- type casting ---
            if self.__f64_act:
                actions = actions.astype(np.float64)
            # --------------------
            
            #^ --- Env step ---
            next_obs, reward, terminated, truncated, info = self.__env.step(actions=actions)
            #^ ----------------
            
            
            #* --- TimeLimit Bootstrap Injection ---
            if np.any(truncated) and "final_observation" in info:
                final_obs_mask = info.get("_final_observation", np.zeros_like(truncated))
                valid_truncation = truncated & final_obs_mask
                
                if np.any(valid_truncation):
                    
                    true_final_obs = info["final_observation"][valid_truncation]
                    if self.__f64_obs: 
                        true_final_obs = true_final_obs.astype(np.float32)
                    
                    norm_final_obs = obs_normalizer.normalize(data=true_final_obs)
                    final_value = self.__critic.forward(norm_final_obs).flatten()
                    reward[valid_truncation] += hyper_params.gamma * final_value
            #* --------------------------------------
            
            
            #^ --- checks if an episode ended ---
            done = terminated | truncated
            #^ ----------------------------------
            
            #~ --- add all the data needed for the optimisation to the buffer ---
            rollout_buffer.add_step(
                observations=norm_obs,
                actions=raw_actions,
                reward=reward,
                done=done,
                means=means,
                stds=stds
            )
            #~ ------------------------------------------------------------------
            
            #? --- update the obs for the next step ---
            obs:np.ndarray = next_obs
            #? ----------------------------------------
        
        #^ --- getting the last observations ---
        #~ the gae calculations needs the last obs of the info
        last_obs = obs.copy()
        #~ but if an episode just terminated, 
        #~ the last obs are the obs from the new states and not the last step.
        #~ But, we can access the real last obs, it's by convenction in the info dict of a vector gym env.
        if "final_observation" in info:
            #? gym gives us a bool mask of wich episodes just ended
            mask = info["_final_observation"]
            #? here, it takes the finals obs of the env
            #? and puts it in the last obs matrix to the right indexes (using the mask)
            last_obs[mask] = info["final_observation"][mask]
        #^ -------------------------------------
        
        #^ --- noramlize the last obs (without updating so theres no bias) ---
        last_norm_obs = obs_normalizer.normalize(data=last_obs)
        #^ -------------------------------------------------------------------
        
        
        
        #~ --- saves the state for the next traj ---
        self.__curent_obs = obs
        self.__curent_info = info
        #~ -----------------------------------------
        
        return last_norm_obs

    
    def __update_networks(self,
                          rollout_buffer: RolloutBuffer,
                          last_norm_obs: np.ndarray,
                          hyper_params: HyperParams) -> Tuple[np.ndarray,np.ndarray,int]:
        
        
        """Internal helper that handles update networks.
        
        Args:
            rollout_buffer (RolloutBuffer): Buffer instance that stores intermediate rollout or.
                                            replay data.
            gae (np.ndarray): Array containing gae values.
            critic_goal (np.ndarray): Array containing critic goal values.
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Computed result expressed as `Tuple[np.ndarray,
                                                np.ndarray, int]`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.__update_networks(rollout_buffer=..., gae=..., critic_goal=..., ...)
        """
        
        #^ --- computes all the values from the critic ---
        #~ we compute them all at the end to minigate the python overhead 
        rollout_buffer.fill_values_buffers(critic=self.__critic)
        #^ -----------------------------------------------
        
        # --- anti out of bound code ---
        if hyper_params.penalize_out_of_action:
            # --- to know how often hes out of bound ---
            violations = np.logical_or(rollout_buffer.actions < self.__low_action, rollout_buffer.actions > self.__high_action)
            violation_rate = violations.sum() / violations.size
            # ------------------------------------------
            
            # --- loss derivative and optim step ---
            d_c_loss = (violation_rate - hyper_params.out_of_action_acceptable_violation_rate)
            self.__out_of_action_optimizer.step(d_c_loss)
            # --------------------------------------
            
            # --- final penalisations ---
            raw_violation = np.maximum(0.0, rollout_buffer.actions - self.__high_action)**2 + np.maximum(0.0, self.__low_action - rollout_buffer.actions)**2
            penalty_value = np.sum(raw_violation, axis=-1)
            rollout_buffer.rewards[:] -= np.exp(self.__log_out_of_action_coeff) * penalty_value
            # ---------------------------
        # -------------------------------
        
        #^ --- computes the gae and gives us the values the critic should've guessed ---
        gae, critic_goal = rollout_buffer.compute_GAE(critic=self.__critic,
                                                      last_obs=last_norm_obs,
                                                      hyper_params=hyper_params)
        #^ -----------------------------------------------------------------------------
        
        
        total_transitions = rollout_buffer.rollout_size * self.__population_size
        num_batches_per_epoch = total_transitions // hyper_params.mini_batch_size
        losses = np.empty(shape=(hyper_params.ppo_epoch * num_batches_per_epoch, 2), dtype=np.float32)
        n_of_iterations = 0
        #! -------------------
        
        #? --- flattening all the epoch data ---
        flat_obs = rollout_buffer.observations.reshape(-1,self.__env.single_observation_space.shape[0])
        flat_actions = rollout_buffer.actions.reshape(-1,self.__env.single_action_space.shape[0])
        flat_log_probs = rollout_buffer.log_probs.flatten()
        flat_gae = gae.flatten()
        flat_critic_goal = critic_goal.flatten()
        #? -------------------------------------
        
        #~ to make sure the std = 1 and mean = 0 for the computed advatage this is way better for stability
        norm_flat_gae = normalize_values(flat_gae)
        
        #^ --- creating the initial indexs ---
        random_indexs = np.arange(rollout_buffer.rollout_size * self.__population_size)
        #^ -----------------------------------
        
        #? We train on the same trajectories multiple times to 
        for epoch in range(hyper_params.ppo_epoch):
            #^ --- randomly shuffles the indexes ---
            #~ to train on non-connected step to break the linear corolation of the steps
            rng.shuffle(random_indexs)
            #^ -------------------------------------
            
            #^ --- shuffles all the info in the same way as random indexs ---
            shuffled_obs = flat_obs[random_indexs]
            shuffled_actions = flat_actions[random_indexs]
            shuffled_gae = norm_flat_gae[random_indexs]
            shuffled_critic_goal = flat_critic_goal[random_indexs]
            shuffled_log_probs = flat_log_probs[random_indexs]
            
            
            for start in range(0,rollout_buffer.rollout_size*self.__population_size,hyper_params.mini_batch_size):
                #^ --- gets the specefics indexs for this mini batch ---
                index = slice(start,start+hyper_params.mini_batch_size)
                obs = shuffled_obs[index]
                actions = shuffled_actions[index]
                log_prob = shuffled_log_probs[index]
                gae = shuffled_gae[index]
                critic_goal = shuffled_critic_goal[index]
                #^ -----------------------------------------------------

                #^ --- does backpropagation to both nn to uptate their params ---
                critic_loss = self.__update_critic(obs=obs, 
                                                   critic_goal=critic_goal,
                                                   hyper_params=hyper_params)
                
                actor_loss = self.__update_actor(obs=obs,
                                                 actions=actions, 
                                                 gae=gae, 
                                                 log_prob=log_prob, 
                                                 hyper_params=hyper_params)
                #^ --------------------------------------------------------------
                
                
                #! -- for testing ---
                losses[n_of_iterations] = critic_loss, actor_loss
                n_of_iterations+=1
                #! ------------------
        
        return losses[:,0], losses[:,1], n_of_iterations
    
    def __update_critic(self, 
                        obs: np.ndarray,
                        critic_goal: np.ndarray,
                        hyper_params: HyperParams)-> float:
        #~ backward does =+ and not = so grads can acumulates so we need to reset them at the start
        """Internal helper that handles update critic.
        
        Args:
            obs (np.ndarray): Observation array used by the routine.
            critic_goal (np.ndarray): Array containing critic goal values.
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
        
        Returns:
            float: Computed result expressed as `float`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.__update_critic(obs=..., critic_goal=..., hyper_params=...)
        """
        self.__critic.reset_grads()
        
        #? make the forward pass to fill the input buffer and get the value predicted
        value = self.__critic.forward(obs)
        
        #~ dimension things
        if critic_goal.ndim == 1:
            critic_goal = critic_goal.reshape(-1, 1)
        
        #^ --- gets the critic loss ---
        #? the loss is not used in the calculations 
        #? but, it's used to monitor optimisation progress
        critic_loss = np.mean((value-critic_goal)**2)
        #^ ----------------------------
        
        #^ --- gets dL/d(mu) ---
        #~ it's divided by the mini batch size because we want the avarage gradiant from this batch
        dZ = 2 * (value-critic_goal) / hyper_params.mini_batch_size
        #^ ---------------------
        
        #^ --- calculate all the partials derivatives of the params ---
        self.__critic.backward(dZ=dZ)
        #^ ------------------------------------------------------------
        
        
        
        #^ --- updates the params with the optimiser ---
        self.__critic_optimizer.step(self.__critic.grads)
        #^ ---------------------------------------------
        
        return float(critic_loss)
    
    def __update_actor(self, 
                       obs: np.ndarray,
                       actions: np.ndarray,
                       gae: np.ndarray,
                       log_prob: np.ndarray,
                       hyper_params: HyperParams)-> float:
        """Internal helper that handles update actor.
        
        Args:
            obs (np.ndarray): Observation array used by the routine.
            actions (np.ndarray): Batch of actions associated with the current transitions.
            gae (np.ndarray): Array containing gae values.
            log_prob (np.ndarray): Array containing log prob values.
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
        
        Returns:
            float: Computed result expressed as `float`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.__update_actor(obs=..., actions=..., gae=..., ...)
        """
        self.__actor.reset_grads()
        ratios, mean, std = self.__prob_ratio(log_prob=log_prob,
                                              obs=obs,
                                              action=actions)
        surr1 = ratios * gae
        surr2 = np.clip(ratios,1-hyper_params.epsilon,1+hyper_params.epsilon) * gae
        entropy = np.sum(np.log(std) + 0.5 * LOG_2_PI + 0.5, axis=-1, keepdims=True)
        actor_loss = np.mean(-np.minimum(surr1, surr2) - hyper_params.entropy_coef * entropy)
        ratio_mask = (surr1 <= surr2)
        dratio:np.ndarray = - gae * ratio_mask
        d_log_prob:np.ndarray = dratio * ratios
        d_log_prob = d_log_prob.reshape(-1,1)
        dmean = d_log_prob * ((actions-mean)/(std**2))
        dstd = d_log_prob * ((actions-mean)**2/(std**3) - 1/std) - hyper_params.entropy_coef / std
        dmean /= obs.shape[0]
        dstd /= obs.shape[0]
        self.__actor.backward(dmean=dmean,dstd=dstd)
        self.__actor_optimizer.step(self.__actor.grads)
        return actor_loss
    
    def __prob_ratio(self, log_prob:np.ndarray, obs: np.ndarray,action: np.ndarray)-> np.ndarray:
        """Internal helper that handles prob ratio.
        
        Args:
            log_prob (np.ndarray): Array containing log prob values.
            obs (np.ndarray): Observation array used by the routine.
            action (np.ndarray): Array containing action values.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.__prob_ratio(log_prob=..., obs=..., action=...)
        """
        mean, std = self.__actor.stochastic_forward(obs)
        return self.__compute_ratio(log_prob,mean,std,action), mean, std
    @staticmethod
    @njit(fastmath=True,cache=True)
    def __compute_ratio(log_prob:np.ndarray, mean:np.ndarray, std:np.ndarray, action:np.ndarray)-> np.ndarray:
        """Internal helper that handles compute ratio.
        
        Args:
            log_prob (np.ndarray): Array containing log prob values.
            mean (np.ndarray): Array containing mean values.
            std (np.ndarray): Array containing std values.
            action (np.ndarray): Array containing action values.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.__compute_ratio(log_prob=..., mean=..., std=..., ...)
        """
        new_log_prob = compute_log_prob(mean,std,action)
        return np.exp(new_log_prob-log_prob)
    
    @override
    class TrainingStats(NetworkOptimiser.TrainingStats):
        """Typed dictionary describing the statistics reported during training.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Attributes:
            score (float): Current score or metric used by the routine.
            actor_loss (np.ndarray): Array containing actor loss values.
            critic_loss (np.ndarray): Array containing critic loss values.
        
        Example:
            >>> obj = TrainingStats()
        """


        actor_loss: np.ndarray
        critic_loss: np.ndarray
    @dataclass
    class HyperParams(NetworkOptimiser.HyperParams):
        """Base marker class for algorithm-specific hyperparameter containers.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Attributes:
            epoch_count (int): Number of epoch used by the routine.
            rollout_length (int): Value supplied for `rollout_length`.
            gamma (float): Discount or exponential-decay factor used by the algorithm.
            lambd (float): GAE smoothing factor controlling the bias-variance trade-off.
            ppo_epoch (int): Value supplied for `ppo_epoch`.
            mini_batch_size (int): Size of the mini batch dimension or buffer.
            epsilon (float): Small constant used to keep divisions numerically stable.
            entropy_coef (float): Value supplied for `entropy_coef`.
            start_epoch (int): Value supplied for `start_epoch`.
            initial_best (float): Value supplied for `initial_best`.
            normalize_obs (bool): Value supplied for `normalize_obs`.
        
        Example:
            >>> obj = HyperParams()
        """
        end_epoch : int = 100
        start_epoch: int = 1
        rollout_length: int = 1024
        gamma: float = 0.99
        lambd: float = 0.90
        ppo_epoch: int = 5
        mini_batch_size: int = 64
        epsilon: float = 0.2
        entropy_coef: float = 0.0
        initial_best: float = float('-inf')
        normalize_obs: bool = True
        penalize_out_of_action: bool = False
        out_of_action_acceptable_violation_rate: float = 0.02
        out_of_action_learning_rate: float = 1.0e-3
        log_out_of_action_coeff: None|np.ndarray = None
    
    def setup_training(self,
                       hyper_params: HyperParams=None,
                       obs_normalizer:Normalizer=None,
                       schedulers : dict[str, scheduler.Scheduler]|None=None) -> None:
        """Prepare the optimizer for training.
        
        Args:
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
            obs_normalizer (Normalizer): Observation normalizer applied before network.
                                            evaluation.
            schedulers (dict[str, scheduler.Scheduler] | None): Named schedulers updated.
                                                                alongside the training loop.
        """
        if schedulers is None:
            self.__schedulers = {}
        else:
            self.__schedulers = schedulers
        # --------------------------

        # --- hyper params init ---
        if hyper_params is None:
            hyper_params = self.HyperParams()
        self.__hyper_params = hyper_params
        # -------------------------
        
        if hyper_params.penalize_out_of_action:
            if hyper_params.log_out_of_action_coeff is not None and not isinstance(hyper_params.log_out_of_action_coeff, np.ndarray):
                raise ValueError("log_out_of_action_coeff needs to be a numpy array if penalize_out_of_action is True")
            if hyper_params.log_out_of_action_coeff == None:
                hyper_params.log_out_of_action_coeff = np.array(np.log(1.0))
            self.__log_out_of_action_coeff = hyper_params.log_out_of_action_coeff
            self.__out_of_action_optimizer = optim.Adam(maximise=True, alpha=hyper_params.out_of_action_learning_rate)
            self.__out_of_action_optimizer.initialize(self.__log_out_of_action_coeff)
        
        # --- calculating rollout size ---
        # Calculate valid total steps across ALL envs (must be multiple of mini_batch_size)
        total_steps = (hyper_params.rollout_length * self.__population_size // hyper_params.mini_batch_size) * hyper_params.mini_batch_size
        # Divide back down to get the exact size per environment
        rollout_size = total_steps // self.__population_size
        if rollout_size == 0:
            raise ValueError("mini batch size can't be larger than the rollout_size")
        # --------------------------------
        
        # --- optimiser initialisation ---
        self.__actor_optimizer.initialize(self.__actor.params,weight_mask=self.__actor.w_mask)
        self.__critic_optimizer.initialize(self.__critic.params,weight_mask=self.__critic.w_mask)
        # --------------------------------
        
        # --- creating the rollout buffer ---
        self.__rollout_buffer = RolloutBuffer(
            obs_dim_count=self.__env.single_observation_space.shape[0],
            action_dim_count=self.__env.single_action_space.shape[0],
            rollout_size=rollout_size,
            population_size=self.__population_size
        )
        # ----------------------------------
        
        # --- observation normalisation handling ---
        if obs_normalizer is not None:
            self.__obs_normalizer = obs_normalizer
        elif hyper_params.normalize_obs:
            self.__obs_normalizer = RunningMeanStd(
                data_length=self.__env.single_observation_space.shape[0]
            )
        else:
            self.__obs_normalizer = NoNormalizer()
        # -----------------------------------------
        self.__training_ready = True
    
    OptimiserCallback = Callable[[TrainingStats, "PPOptimiser"],bool]
    @override
    def train(self,
              callback: Optional[OptimiserCallback] = None
              ) -> None:
        """Train the nn on the environnement

        Args:
            callback (Optional[OptimiserCallback], optional): a fonction that has in Args the training stats and an ppotimiser . Defaults to None.
        """
        if callback is not None and not callable(callback):
            raise ValueError("callback needs to be a callable function")
        if not self.__training_ready:
            raise nn.InitializationException("you need to call setup_training before calling train")
        hyper_params = self.__hyper_params
        if self.hyper_params.start_epoch > self.hyper_params.end_epoch:
            print_red_warning_simple("Tried start epoch > end epoch so no training happend")
            return
        
        # --- stats variable creation ---
        last_known_score = 0.0
        total_param_update = 0
        n_of_steps = 0
        training_starting_time = time.perf_counter()
        # -------------------------------
        
        #^ --- gets the first obs of the env ---
        self.__curent_obs, self.__curent_info = self.__env.reset(seed=int(rng.integers(0,100_000_000)))
        #^ -------------------------------------
        
        for i in range(hyper_params.start_epoch, hyper_params.end_epoch + 1):
            
            # --- time variable ---
            traj_start_time = time.perf_counter()
            # ---------------------
            
            last_norm_obs = self.__get_trajectories(
                rollout_buffer=self.__rollout_buffer,
                obs_normalizer=self.__obs_normalizer,
                hyper_params=hyper_params
            )
            
            # --- calculates the mean score before the out of action loss ---
            score = self.__rollout_buffer.get_mean_score()
            # ---------------------------------------------------------------
            
            # --- time variable ---
            traj_dt = time.perf_counter() - traj_start_time
            params_update_start_time = time.perf_counter()
            # ---------------------
            
            critic_loss, actor_loss, n_of_iterations = self.__update_networks(rollout_buffer=self.__rollout_buffer,
                                                                              last_norm_obs=last_norm_obs,
                                                                              hyper_params=hyper_params)
            
            # --- time variable ---
            params_update_dt = time.perf_counter() - params_update_start_time
            # ---------------------
            
            
            if score is not None:
                last_known_score = score
            
            # --- update the schedulers ---
            for sched in self.__schedulers:
                self.__schedulers[sched].step(last_known_score)
            # -----------------------------
            
            # --- stats update ---
            n_of_steps += self.__rollout_buffer.rollout_size * self.__env.num_envs
            total_param_update += n_of_iterations
            total_training_time = time.perf_counter() - training_starting_time
            # --------------------
            
            #^ --- calling the callback ---
            if callback is not None:
                stats_dict: PPOptimiser.TrainingStats = {
                    "current_epoch":i,
                    "number_of_steps":n_of_steps,
                    "total_param_update":total_param_update,
                    "params_updating_dt":params_update_dt,
                    "trajectories_collecting_dt":traj_dt,
                    "total_training_time":total_training_time,
                    "score":last_known_score,
                    "actor_loss":actor_loss,
                    "critic_loss":critic_loss,
                }
                training_ended = callback(stats_dict, self)
                if training_ended:
                    print("training stoped by callback")
                    break
            #^ ---------------------------
    @override
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through the actor network.
        
        Args:
            obs (np.ndarray): Observation array used by the routine.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        
        Example:
            >>> obj = PPOptimiser(...)
            >>> result = obj.forward(obs=...)
        """
        if not self.__training_ready:
            raise nn.InitializationException("you need to call setup_training before calling forward")
        if self.__obs_normalizer is not None:
            obs = self.__obs_normalizer.normalize(obs)
        return self.__actor.deterministic_forward(obs)
    @property
    def log_out_of_action_coeff(self):
        """Return the current log out of action coeff.
        
        Returns:
            Any: Computed result returned by the function.
        
        Raises:
            ValueError: Raised if the component is not configured to penalize out-of-action steps.
        """
        if not hasattr(self, "_PPOptimiser__log_out_of_action_coeff"):
            return None
        return self.__log_out_of_action_coeff
    @log_out_of_action_coeff.setter
    def log_out_of_action_coeff(self, value: float|np.ndarray):
        """Sets the log out of action coeff.
        
        Args:
            value (float): The new value for the log out of action coeff.
        
        Raises:
            ValueError: Raised if the component is not configured to penalize out-of-action steps.
        """
        if not hasattr(self, "_PPOptimiser__log_out_of_action_coeff"):
            raise ValueError("this PPO instance is not configured to penalize out-of-action steps")
        if not isinstance(value, (float, np.ndarray)):
            raise ValueError(f"log_out_of_action_coeff must be a float or a numpy array got {type(value)}")
        if isinstance(value, float):
            value = np.array(value, dtype=np.float32)
        self.__log_out_of_action_coeff = value.astype(np.float32)
        self.__out_of_action_optimizer.initialize(self.__log_out_of_action_coeff)
    @property
    def out_of_action_optimizer(self):
        """Return the current out of action optimizer.
        
        Returns:
            Any: Computed result returned by the function.
        
        Raises:
            ValueError: Raised if the component is not configured to penalize out-of-action steps.
        """
        if not hasattr(self, "_PPOptimiser__out_of_action_optimizer"):
            return None
        return self.__out_of_action_optimizer
    @out_of_action_optimizer.setter
    def out_of_action_optimizer(self, value: optim.GradientOptimizer):
        """Sets the out of action optimizer.
        
        Args:
            value (Optimizer): The optimizer instance to use for updating the out of action coeff.
        
        Raises:
            ValueError: Raised if the component is not configured to penalize out-of-action steps.
        """
        if not hasattr(self, "_PPOptimiser__out_of_action_optimizer"):
            raise ValueError("this PPO instance is not configured to penalize out-of-action steps")
        if not isinstance(value, optim.GradientOptimizer):
            raise ValueError(f"out_of_action_optimizer must be an instance of Optimizer got {type(value)}")
        self.__out_of_action_optimizer = value
        
    @property
    def out_of_action_coeff(self):
        """Return the current out of action coeff.
        
        Returns:
            Any: Computed result returned by the function.
        
        Raises:
            ValueError: Raised if the component is not configured to penalize out-of-action steps.
        """
        if not hasattr(self, "_PPOptimiser__log_out_of_action_coeff"):
            return None
        return float(np.exp(self.__log_out_of_action_coeff))
    @property
    def actor(self):
        """Return the current actor.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__actor
    @property
    def critic(self):
        """Return the current critic.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__critic
    @property
    def env(self):
        """Return the current env.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__env
    @property
    def actor_optimizer(self):
        """Return the current actor optimizer.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__actor_optimizer
    @property
    def critic_optimizer(self):
        """Return the current critic optimizer.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__critic_optimizer
    @property
    def schedulers(self):
        """Return the current schedulers.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__schedulers
    @property
    def hyper_params(self):
        """Return the current hyper parameters."""
        return self.__hyper_params
    @property
    def population_size(self):
        """Return the current population size.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__population_size
    @property
    def obs_normalizer(self):
        """Return the current obs normalizer.
        
        Returns:
            Any: Computed result returned by the function.
        
        Raises:
            SimTorch.core.nn.InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        try :
            return self.__obs_normalizer
        except :
            raise nn.InitializationException("there's no training done on this PPO instance, so theres still no normalizer")
    @obs_normalizer.setter
    def obs_normalizer(self, value: Normalizer):
        """Sets the observation normalizer.
        
        Args:
            value (Normalizer): The normalizer instance to use.
        """
        if not isinstance(value, Normalizer):
            raise ValueError(f"the obs_normalizer must be an instance of Normalizer got {type(value)}")
        self.__obs_normalizer = value


