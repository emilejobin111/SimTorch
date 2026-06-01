"""Soft Actor-Critic implementation for continuous-control training.

This module provides the replay buffer, target-network updates, entropy tuning, and
training loop required to train SAC policies in vectorized Gymnasium environments.
"""
# --- base python import ---
from __future__ import annotations
from typing import List, override, Tuple, Any, Callable, Optional
from dataclasses import dataclass
import time
from copy import deepcopy
# --------------------------


# --- ext lib imports ---
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
# ------------------------

# --- SimTorch.core imports ---
from ..core import nn
from ..core import optim
from ..core import scheduler
from ..core.rng import rng
# -----------------------------


from ..save.save_nn import CHECKPOINT_DIR, write_metadata, read_metadata


# --- SimTorch.rl imports ---
from . import policy
from .utils import (Normalizer,
                    NoNormalizer,
                    RunningMeanStd,
                    LOG_2_PI,
                    normalize_values,
                    compute_tanh_log_prob,
                    print_red_warning_simple)
from .base_nn_optimiser import NetworkOptimiser
# -----------------------------


class SACRolloutBuffer:
    """A cyclic replay buffer used for storing and sampling experience transitions for SAC.
    
    Instances preallocate NumPy-backed storage so training loops can reuse memory
    efficiently across updates.
    
    Args:
        obs_dim_count (int): Number of scalar features in one observation.
        action_dim_count (int): Number of scalar features in one action.
        rollout_size (int): Maximum number of transitions stored by the buffer.
    
    Example:
        >>> obj = SACRolloutBuffer(obs_dim_count=4, action_dim_count=4, rollout_size=4)
    """




    def __init__(self, obs_dim_count:int, action_dim_count:int, rollout_size:int=1_000_000,):
        """Initializes the SACRolloutBuffer with pre-allocated numpy arrays.
        
        Args:
            obs_dim_count (int): The dimensionality of the observation space.
            action_dim_count (int): The dimensionality of the action space.
            rollout_size (int): The maximum number of steps the buffer can store. Defaults to.
                                1,000,000.
        """
        self.__rollout_size = rollout_size
        self.__index = 0
        self.__obs = np.zeros((rollout_size, obs_dim_count), dtype=np.float32)
        self.__actions = np.zeros((rollout_size, action_dim_count), dtype=np.float32)
        self.__next_obs = np.zeros_like(self.__obs)
        self.__rewards = np.zeros((rollout_size), dtype=np.float32)
        self.__dones = np.zeros((rollout_size), dtype=np.bool_)

    @property
    def rollout_size(self) -> int:
        """Retrieves the maximum capacity of the rollout buffer.
        
        Returns:
            int: Computed result expressed as `int`.
        """
        return self.__rollout_size

    @property
    def index(self) -> int:
        """Retrieves the total number of transitions added to the buffer so far.
        
        Returns:
            int: Computed result expressed as `int`.
        """
        return self.__index
    @index.setter
    def index(self, value:int):
        """Sets the current index of the buffer, which tracks how many transitions have been added.
        
        Args:
            value (int): The new index value to set. Should be a non-negative integer.
        """
        if not isinstance(value, int):
            raise TypeError("Index must be an integer.")
        if value < 0:
            raise ValueError("Index must be a non-negative integer.")
        self.__index = value

    @property
    def obs(self) -> np.ndarray:
        """Retrieves the array containing all stored observations.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        """
        return self.__obs
    @obs.setter
    def obs(self, value:np.ndarray):
        """Sets the observation array of the buffer.
        
        Args:
            value (np.ndarray): The new observation array to set. Should have shape (rollout_size, obs_dim_count).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Observations must be a numpy array.")
        self.__obs = value

    @property
    def actions(self) -> np.ndarray:
        """Retrieves the array containing all stored actions.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        """
        return self.__actions
    @actions.setter
    def actions(self, value:np.ndarray):
        """Sets the action array of the buffer.
        
        Args:
            value (np.ndarray): The new action array to set. Should have shape (rollout_size, action_dim_count).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Actions must be a numpy array.")
        self.__actions = value

    @property
    def next_obs(self) -> np.ndarray:
        """Retrieves the array containing all stored next observations.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        """
        return self.__next_obs
    @next_obs.setter
    def next_obs(self, value:np.ndarray):
        """Sets the next observation array of the buffer.
        
        Args:
            value (np.ndarray): The new next observation array to set. Should have shape (rollout_size, obs_dim_count).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Next observations must be a numpy array.")
        self.__next_obs = value

    @property
    def rewards(self) -> np.ndarray:
        """Retrieves the array containing all stored rewards.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        """
        return self.__rewards
    @rewards.setter
    def rewards(self, value:np.ndarray):
        """Sets the rewards array of the buffer.
        
        Args:
            value (np.ndarray): The new rewards array to set. Should have shape (rollout_size,).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Rewards must be a numpy array.")
        self.__rewards = value
    @property
    def dones(self) -> np.ndarray:
        """Retrieves the array containing all stored done flags.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        """
        return self.__dones
    @dones.setter
    def dones(self, value:np.ndarray):
        """Sets the done flags array of the buffer.
        
        Args:
            value (np.ndarray): The new done flags array to set. Should have shape (rollout_size,).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Dones must be a numpy array.")
        self.__dones = value

    def add_step(self,
                  observations:np.ndarray,
                  actions:np.ndarray,
                  next_observations:np.ndarray,
                  rewards:np.ndarray,
                  dones:np.ndarray):
        """Adds a batch of experience steps to the buffer, automatically handling cyclic wrap-around.
        
        The method operates on preallocated storage so experience collection can proceed without
        repeated memory allocation.
        
        Args:
            observations (np.ndarray): Array of current observations. Shape: (num_steps,.
                                       obs_dim_count).
            actions (np.ndarray): Array of actions taken. Shape: (num_steps, action_dim_count).
            next_observations (np.ndarray): Array of resulting next observations. Shape:.
                                            (num_steps, obs_dim_count).
            rewards (np.ndarray): Array of rewards received. Shape: (num_steps,).
            dones (np.ndarray): Array of boolean done flags indicating episode termination.
        
        Returns:
            None: None.
        
        Example:
            >>> obj = SACRolloutBuffer(...)
            >>> result = obj.add_step(observations=..., actions=..., next_observations=..., ...)
        """
        num_steps = len(dones)
        start_idx = self.__index % self.rollout_size
        end_idx_linear = start_idx + num_steps
        
        # Determine if wrap-around occurs
        if end_idx_linear <= self.rollout_size:
            # Case 1: No wrap-around
            end_slice = start_idx + num_steps
            self.__obs[start_idx:end_slice] = observations
            self.__actions[start_idx:end_slice] = actions
            self.__next_obs[start_idx:end_slice] = next_observations
            self.__rewards[start_idx:end_slice] = rewards
            self.__dones[start_idx:end_slice] = dones
        else:
            # Case 2: Wrap-around occurs
            # Part 1: From start_idx to end of buffer
            steps_part1 = self.rollout_size - start_idx
            self.__obs[start_idx:] = observations[:steps_part1]
            self.__actions[start_idx:] = actions[:steps_part1]
            self.__next_obs[start_idx:] = next_observations[:steps_part1]
            self.__rewards[start_idx:] = rewards[:steps_part1]
            self.__dones[start_idx:] = dones[:steps_part1]
            
            # Part 2: From beginning of buffer (index 0)
            steps_part2 = num_steps - steps_part1
            self.__obs[:steps_part2] = observations[steps_part1:]
            self.__actions[:steps_part2] = actions[steps_part1:]
            self.__next_obs[:steps_part2] = next_observations[steps_part1:]
            self.__rewards[:steps_part2] = rewards[steps_part1:]
            self.__dones[:steps_part2] = dones[steps_part1:]
            
        self.__index += num_steps

    def sample_random(self, batch_size:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a random batch of experience transitions from the populated portion of the buffer.
        
        The method operates on preallocated storage so experience collection can proceed without
        repeated memory allocation.
        
        Args:
            batch_size (int): The number of transitions to sample.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple
                                                                               containing
                                                                               batches of
                                                                               (observations,
                                                                               actions,
                                                                               next_observations,
                                                                               rewards, dones).
                                                                               Rewards and dones
                                                                               are reshaped to
                                                                               (-1, 1).
        
        Raises:
            ValueError: Raised if one or more argument values are invalid or unsupported.
        
        Example:
            >>> obj = SACRolloutBuffer(...)
            >>> result = obj.sample_random(batch_size=4)
        """
        if self.__index < batch_size:
            raise ValueError(f"Not enough data to sample. Current index: {self.__index}, requested batch size: {batch_size}")
        max_index = min(self.__index, self.rollout_size)
        indices = rng.integers(0, max_index, size=batch_size)
        return (np.ascontiguousarray(self.__obs[indices]),
                np.ascontiguousarray(self.__actions[indices]),
                np.ascontiguousarray(self.__next_obs[indices]),
                np.ascontiguousarray(self.__rewards[indices].reshape(-1, 1)),
                np.ascontiguousarray(self.__dones[indices].reshape(-1, 1)))


class SACOptimiser(NetworkOptimiser):
    """An optimiser class that implements the Soft Actor-Critic (SAC) reinforcement learning algorithm.
    
    The implementation coordinates runtime state, optimization steps, and
    checkpoint-friendly serialization helpers needed by the training workflow.
    
    Args:
        env (gym.vector.VectorEnv): Gymnasium environment used for interaction or.
                                    validation.
        actor_network (policy.ActorNN): Actor network that produces policy outputs.
        q_networks (List[policy.QNN]): Collection of Q networks used by the SAC objective.
        actor_optimizer (optim.GradientOptimizer): Optimizer used to update the actor.
                                                   network.
        q_optimizers (List[optim.GradientOptimizer]): Optimizers used to update the Q.
                                                      networks.
    
    Example:
        >>> obj = SACOptimiser(env=env, actor_network=..., q_networks=..., ...)
    """




    def __init__(self,
                 env:gym.vector.VectorEnv,
                 actor_network:policy.ActorNN,
                 q_networks:List[policy.QNN],
                 actor_optimizer:optim.GradientOptimizer,
                 q_optimizers:List[optim.GradientOptimizer],
                 rollout_buffer:Optional[SACRolloutBuffer]=None
                 )->None:
        """Initializes the SACOptimiser with the environment, networks, and individual optimizers.
        
        Args:
            env (gym.vector.VectorEnv): The vectorized gymnasium environment to train on.
            actor_network (policy.ActorNN): The actor network responsible for policy selection.
            q_networks (List[policy.QNN]): A list containing exactly two Q-networks (critics).
            actor_optimizer (optim.GradientOptimizer): The gradient optimizer for the actor.
                                                       network.
            q_optimizers (List[optim.GradientOptimizer]): A list containing the gradient.
                                                          optimizers for the Q-networks.
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            NotImplementedError: Raised if the requested configuration is not implemented.
        """
        
        # --- type checking ---
        if not isinstance(env, gym.vector.VectorEnv):
            raise TypeError(f"env must be an instance of gym.vector.VectorEnv, got {type(env).__name__}")
        if not isinstance(actor_network, policy.ActorNN):
            raise TypeError(f"actor_network must be an instance of policy.ActorNN, got {type(actor_network).__name__}")
        if not isinstance(q_networks, (list, tuple)) or len(q_networks) != 2 or not all(isinstance(q, policy.QNN) for q in q_networks):
            raise TypeError("q_networks must be a list or tuple of two instances of policy.QNN")
        if not isinstance(actor_optimizer, optim.GradientOptimizer):
            raise TypeError(f"actor_optimizer must be an instance of optim.GradientOptimizer, got {type(actor_optimizer).__name__}")
        if not isinstance(q_optimizers, (list, tuple)) or len(q_optimizers) != 2 or not all(isinstance(opt, optim.GradientOptimizer) for opt in q_optimizers):
            raise TypeError("q_optimizers must be a list or tuple of two instances of optim.GradientOptimizer")
        if not isinstance(env.single_action_space, gym.spaces.Box):
            raise NotImplementedError("SAC only works with continuous actions spaces")
        if rollout_buffer is not None and not isinstance(rollout_buffer, SACRolloutBuffer):
            raise TypeError(f"rollout_buffer must be an instance of SACRolloutBuffer or None, got {type(rollout_buffer).__name__}")
        # ----------------------
        else :
            self.__low_action, self.__high_action = env.single_action_space.low, env.single_action_space.high
        if rollout_buffer is None:
            self.__rollout_buffer = None
        else:
            self.__rollout_buffer = rollout_buffer
        self.__f64_obs = env.single_observation_space.dtype == np.float64
        self.__f64_act = env.single_action_space.dtype == np.float64
        self.__obs_dim = env.single_observation_space.shape[0]
        self.__action_dim = env.single_action_space.shape[0]
        self.__env = env
        self.__population_size = env.num_envs
        self.__pi_optimizer = actor_optimizer
        self.__q_optimizers = q_optimizers
        
        self.__pi = actor_network
        self.__qs = q_networks
        self.__q_tarrs = [deepcopy(q_networks[0]), deepcopy(q_networks[1])]

        # --- for the callback info dict ---
        self.__n_of_params_update = 0
        # ----------------------------------
        
        self.__training_ready = False
        
    
    def __get_steps(self,
                           rollout_buffer:SACRolloutBuffer,
                           obs_normalizer: Normalizer,
                           hyper_params: HyperParams,
                           step_amount:int=None,
                           random_actions:bool=False) -> None:
        """Executes environment steps to collect trajectory data and stores it in the rollout buffer.
        
        Args:
            rollout_buffer (SACRolloutBuffer): The buffer to store the collected transitions.
            obs_normalizer (Normalizer): The normalization utility for raw observations.
            hyper_params (HyperParams): The hyperparameter configuration object.
            step_amount (int): The number of individual steps to collect. Defaults to None.
            random_actions (bool): If True, takes random uniform actions for exploration.
                                   Defaults to False.
        
        Returns:
            None: None.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.__get_steps(rollout_buffer=..., obs_normalizer=..., hyper_params=..., ...)
        """
        
        if hyper_params.step_amount % self.__population_size != 0:
            if hyper_params.step_amount < self.__population_size:
                raise ValueError("step_amount should be bigger or equal to the population size")
            last_step_amount = hyper_params.step_amount
            hyper_params.step_amount = (hyper_params.step_amount//self.__population_size) * (self.__population_size+1)
            #the +1 is to make sure that the data to update ratio stays under 1
            print_red_warning_simple(f"step_amount modulo self.__population_size should equal to zero or else the step amount will be and it's now changed to {hyper_params.step_amount} from {last_step_amount}")
        
        if step_amount is None:
            step_amount = hyper_params.step_amount
        
        
        #~ --- gets the curent states ---
        obs = self.__curent_obs
        info = self.__curent_info
        #~ ------------------------------
        
        
        for i in range(step_amount//self.__population_size):
            # --- type casting ---
            if self.__f64_obs:
                obs = obs.astype(np.float32)
            # --------------------
            
            #^ --- normalizes obs ---
            norm_obs = obs_normalizer.normalize_and_update(obs)
            #^ ----------------------
            
            if random_actions:
                actions = rng.random(size=(self.__population_size, self.__env.single_action_space.shape[0])) * (self.__high_action - self.__low_action) + self.__low_action
            else:
                #^ --- gets the action normal distr for the action ---
                means, stds = self.__pi.stochastic_forward(norm_obs)
                #^ ---------------------------------------------------
                
                #^ --- randomly chose the next actions ---
                raw_actions = means + stds * rng.normal(0,1,size=stds.shape)
                #^ --------------------------------------
                
                #~ --- limits the actions from -1 to 1  ---
                actions = np.tanh(raw_actions)
                #~ ----------------------------------------
            
            # --- type casting ---
            if self.__f64_act:
                actions:np.ndarray = actions.astype(np.float64)
            # --------------------
            
            #^ --- Env step ---
            next_obs, reward, terminated, truncated, info = self.__env.step(actions=actions)
            #^ ----------------
            
            #^ --- reward scaling ---
            reward *= hyper_params.reward_scale
            #^ ----------------------
            
            
            #^ --- checks if an episode ended ---
            done = terminated
            #^ ----------------------------------
            
            #^ --- getting the next step observations ommited ---
            #~ SAC needs the last obs of the info
            next_step_obs:np.ndarray
            next_step_obs = next_obs.copy()
            #~ but if an episode just terminated, 
            #~ the last obs are the obs from the new states and not the last step.
            #~ But, we can access the real last obs, it's by convenction in the info dict of a vector gym env.
            if "final_observation" in info:
                #? gym gives us a bool mask of wich episodes just ended
                mask = info["_final_observation"]
                #? here, it takes the finals obs of the env
                #? and puts it in the last obs matrix to the right indexes (using the mask)
                next_step_obs[mask] = info["final_observation"][mask]
            #^ --------------------------------------------------
            
            rollout_buffer.add_step(observations=obs,
                                   actions=actions,
                                   next_observations=next_step_obs,
                                   rewards=reward,
                                   dones=done)
            
            #? --- update the obs for the next step ---
            obs:np.ndarray = next_obs
            #? ----------------------------------------
        self.__curent_obs = obs
        self.__curent_info = info
    
    
    def __get_q_target_value(self,
                           action: np.ndarray,
                           next_obs: np.ndarray,
                           reward: np.ndarray,
                           dones: np.ndarray,
                           hyper_params: HyperParams)->np.ndarray:
        """Computes the target Q-value (Bellman backup) using the target networks and entropy regularization.
        
        Args:
            action (np.ndarray): The batch of actions currently evaluated.
            next_obs (np.ndarray): The batch of next state observations.
            reward (np.ndarray): The batch of immediate rewards.
            dones (np.ndarray): The batch of terminal state flags.
            hyper_params (HyperParams): The hyperparameter configuration object.
        
        Returns:
            np.ndarray: The computed Q-target array.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.__get_q_target_value(action=..., next_obs=..., reward=..., ...)
        """
        
        #^ --- gets the next action ---
        epsillons = rng.normal(size=(action.shape)).astype(np.float32)
        means, stds = self.__pi.stochastic_forward(next_obs)
        next_raw_action = means + stds * epsillons
        next_action = np.tanh(next_raw_action)
        #^ ----------------------------
        
        #^ --- gets the logp of the next_action ---
        log_prop = compute_tanh_log_prob(means=means,
                                         stds=stds,
                                         actions=next_action,
                                         raw_actions=next_raw_action)
        #^ -----------------------------------------
        
        #^ --- reshape the data ---
        log_prop = log_prop.reshape(-1,1)
        #^ ------------------------
        
        
        #^ --- gets the Q tarr values ---
        next_state_action = np.concatenate([next_obs, next_action], axis=-1)
        Q_tarr_1 = self.__q_tarrs[0].forward(next_state_action)
        Q_tarr_2 = self.__q_tarrs[1].forward(next_state_action)
        # Q_tarr_min = np.min(np.array([Q_tarr_1, Q_tarr_2]), axis=0) 
        # my implementation but gemini told me minimum is just better
        Q_tarr_min = np.minimum(Q_tarr_1,Q_tarr_2)
        #^ -------------------------------
        
        #^ --- gets the target for the Q networks ---
        Q_tarr = reward + hyper_params.gamma * (1-dones) * (Q_tarr_min - np.exp(hyper_params.log_alpha)*log_prop)
        #^ ------------------------------------------
        return Q_tarr
    
    
    def __update_q_networks(self,
                            obs: np.ndarray,
                            action: np.ndarray,
                            next_obs: np.ndarray,
                            reward: np.ndarray,
                            dones: np.ndarray,
                            hyper_params: HyperParams)->Tuple[np.ndarray]:
        """Performs a forward pass, loss calculation, backward pass, and optimizer step for both Q-networks.
        
        Args:
            obs (np.ndarray): The batch of current observations.
            action (np.ndarray): The batch of taken actions.
            next_obs (np.ndarray): The batch of next observations.
            reward (np.ndarray): The batch of rewards received.
            dones (np.ndarray): The batch of done flags.
            hyper_params (HyperParams): The hyperparameter configuration object.
        
        Returns:
            Tuple[np.ndarray]: A tuple containing the mean squared error loss for each
                               Q-network.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.__update_q_networks(obs=..., action=..., next_obs=..., ...)
        """
        
        #^ --- Finding the target for the Q networks ---
        # y in the equations is now the Q_tarr
        Q_tarr = self.__get_q_target_value(action=action,
                                           next_obs=next_obs,
                                           reward=reward,
                                           dones=dones,
                                           hyper_params=hyper_params)
        #^ -------------------------------------------------
        
        #^ --- gets the right shape ---
        Q_tarr = Q_tarr.reshape(-1,1)
        #^ ----------------------------
        
        #^ --- compute the losses ---
        state_action = np.concatenate([obs, action], axis=-1)
        losses = []
        values = []
        for q_networks in self.__qs:
            value = q_networks.forward(state_action)
            values.append(value)
            loss = np.mean(0.5 * (value-Q_tarr)**2)
            losses.append(loss)
        #^ -------------------------
        
        #^ --- compute the derivatives ---
        for i in range(len(self.__qs)):
            self.__qs[i].reset_grads()
            dZ = (values[i]-Q_tarr) / len(values[i])
            self.__qs[i].backward(dZ=dZ)
        #^ --------------------------------
        
        #^ --- update the parameters ---
        for i in range(len(self.__qs)):
            self.__q_optimizers[i].step(self.__qs[i].grads)
        #^ -----------------------------
        return losses
    
    
    def __update_pi_and_alpha(self,
                              obs: np.ndarray,
                              act: np.ndarray,
                              hyper_params: HyperParams)->float:
        """Updates the policy network parameters and auto-tunes the entropy temperature coefficient (alpha).
        
        Args:
            obs (np.ndarray): The batch of current observations.
            act (np.ndarray): The batch of original actions (not strictly used directly for.
                              policy update).
            hyper_params (HyperParams): The hyperparameter configuration object.
        
        Returns:
            float: A tuple containing the policy loss and the alpha loss.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.__update_pi_and_alpha(obs=..., act=..., hyper_params=...)
        """
        # just so we don't have to call np.exp every time we use alpha, also cleaner
        alpha = np.exp(hyper_params.log_alpha)
        
        #^ --- gets the actor action ---
        epsillons = rng.normal(size=(act.shape)).astype(np.float32)
        means, stds = self.__pi.stochastic_forward(obs)
        raw_action = means + stds * epsillons
        action = np.tanh(raw_action)
        #^ ------------------------------
        
        #^ --- gets the logp of the next_action ---
        log_prop = compute_tanh_log_prob(means=means,
                                         stds=stds,
                                         actions=action,
                                         raw_actions=raw_action)
        #^ -----------------------------------------
        
        #^ --- reshape the data ---
        log_prop = log_prop.reshape(-1,1)
        #^ ------------------------
        
        #^ --- gets the values predicted ---
        # FIX: Concatenate obs and action before the forward pass
        state_action = np.concatenate([obs, action], axis=-1)
        values = []
        for i in range(len(self.__qs)):
            values.append(self.__qs[i].forward(state_action))
            
        values = np.array(values) # Shape: (2, batch_size, 1)
        
        # FIX: Reshape to 1D array to prevent broadcasting explosion
        min_values_idx = np.argmin(values, axis=0).reshape(-1) # Shape: (batch_size,)
        batch_indices = np.arange(obs.shape[0])
        
        # FIX: Index properly with both dimensions
        min_values = values[min_values_idx, batch_indices] # Shape: (batch_size, 1)
        #^ ---------------------------------
        
        #^ --- compute the loss ---
        pi_loss = np.mean(alpha * log_prop - min_values)
        #^ ------------------------
        
        #^ --- gets the partial derivative from the critics ---
        das = []
        for i in range(len(self.__qs)):
            # FIX: Pass ones to get the pure Jacobian dQ/da
            da = self.__qs[i].backward(np.ones_like(values[i]))[:, self.__obs_dim::]
            das.append(da)
            
        das = np.array(das) # Shape: (2, batch_size, action_dim)
        
        # FIX: Array indices now align correctly -> Shape: (batch_size, action_dim)
        min_das = das[min_values_idx, batch_indices]
        #^ -----------------------------------------------------
        
        #^ --- tanh distr correction ---
        mean_da = (min_das) * (1-action*action)
        std_da = (min_das) * (1-action*action) * epsillons
        #^ -----------------------------
        
        #^ --- log prob derivative ---
        mean_dlogp = alpha * (2 * action)
        safe_std = np.maximum(stds, 1e-8)
        std_dlogp = alpha * (2 * action*epsillons - 1/safe_std)
        #^----------------------------
        
        #^ --- gets the finals derivatives --- 
        dmean = mean_dlogp - mean_da
        dstd = std_dlogp - std_da
        #^ -----------------------------------
        
        #^ --- avarage gradient over the batch ---
        dmean /= obs.shape[0]
        dstd /= obs.shape[0]
        #^ ---------------------------------------
        
        
        
        #^ --- retropropagation ---
        self.__pi.reset_grads()
        self.__pi.backward(dmean=dmean,dstd=dstd)
        #^ ------------------------
        
        #^ --- update the parameters ---
        self.__pi_optimizer.step(self.__pi.grads)
        #^ -----------------------------
        
        if hyper_params.trainable_alpha:
            alpha_loss = - np.mean(alpha * (log_prop + hyper_params.tarr_entropy))
            dalpha = - np.mean(log_prop + hyper_params.tarr_entropy)
            self.__alpha_optimizer.step(dalpha)
        else :
            alpha_loss = 0.0
        return pi_loss, alpha_loss
    
    def __update_tarr_networks(self,
                               hyper_params: HyperParams,) -> None:
        """Applies a soft polyak averaging update to the target Q-networks using the main Q-networks.
        
        Args:
            hyper_params (HyperParams): The hyperparameter configuration object containing the.
                                        'tau' parameter.
        
        Returns:
            None: None.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.__update_tarr_networks(hyper_params=...)
        """
        for i in range(len(self.__q_tarrs)):
            new_params = (1 - hyper_params.tau) * self.__q_tarrs[i].params + hyper_params.tau * self.__qs[i].params
            self.__q_tarrs[i].params_copy(new_params)
    
    def __update_networks(self,
                          rollout_buffer:SACRolloutBuffer,
                          obs_normalizer: Normalizer,
                          hyper_params: HyperParams,
                          step_amount: int=None):
        """Samples a batch from the buffer and coordinates the updates for all networks (Q, Policy, Target).
        
        Args:
            rollout_buffer (SACRolloutBuffer): The memory buffer to sample experience from.
            obs_normalizer (Normalizer): The normalization utility applied to sampled.
                                         observations.
            hyper_params (HyperParams): The hyperparameter configuration object.
            step_amount (int): The number of update iterations to execute. Defaults to None.
        
        Returns:
            Any: A tuple comprising (q_losses, pi_loss, alpha_loss).
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.__update_networks(rollout_buffer=..., obs_normalizer=..., hyper_params=..., ...)
        """
        if step_amount is None:
            step_amount = hyper_params.step_amount
        for i in range(step_amount):
            obs, act, next_obs, reward, dones = rollout_buffer.sample_random(hyper_params.batch_size)
            
            # --- renormalise obs ---
            obs = obs_normalizer.normalize(obs)
            next_obs = obs_normalizer.normalize(next_obs)
            # -----------------------
            
            q_losses = self.__update_q_networks(obs=obs,
                                                action=act,
                                                next_obs=next_obs,
                                                reward=reward,
                                                dones=dones,
                                                hyper_params=hyper_params)
            
            pi_loss, alpha_loss = self.__update_pi_and_alpha(obs=obs,
                                                             act=act,
                                                             hyper_params=hyper_params)
            
            self.__update_tarr_networks(hyper_params=hyper_params)
            
            # --- for the callback info ---
            self.__n_of_params_update += 1
            # -----------------------------
        
        return q_losses, pi_loss, alpha_loss
    
    def test_pi(self,
                obs_normalizer: Normalizer,
                run_amount:int=1):
        """Evaluates the current deterministic policy over a given number of environment episodes.
        
        Args:
            obs_normalizer (Normalizer): The observation normalization utility.
            run_amount (int): Number of episodes to run for the test. Defaults to 1.
        
        Returns:
            Any: The mean accumulated reward across the test episodes.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.test_pi(obs_normalizer=..., run_amount=...)
        """
        
        #^ --- Setting up the vars ---
        score = np.zeros(shape=(run_amount,self.__population_size))
        done = np.zeros(shape=(run_amount,self.__population_size),dtype=np.bool_)
        #^ ---------------------------
        
        for i in range(run_amount):
            
            #^ --- reseting the state with the right env seed ---
            obs, info = self.__env.reset(seed=int(rng.integers(0,100_000_000)))
            #^ --------------------------------------------------
            
            while not done[i].all():
                # --- type casting ---
                if self.__f64_obs:
                    obs = obs.astype(np.float32)
                # --------------------

                #^ --- normalizes obs ---
                norm_obs = obs_normalizer.normalize(obs)
                #^ ----------------------
                
                #^ --- gets the action normal distr for the action ---
                raw_actions = self.__pi.deterministic_forward(norm_obs)
                #^ ---------------------------------------------------
                
                
                #~ --- limits the actions from -1 to 1  ---
                actions = np.tanh(raw_actions)
                #~ ----------------------------------------
                
                # --- type casting ---
                if self.__f64_act:
                    actions:np.ndarray = actions.astype(np.float64)
                # --------------------

                #^ --- Env step ---
                obs, reward, terminated, truncated, info = self.__env.step(actions=actions)
                #^ ----------------

                score[i] += (~done[i]) * reward

                #^ --- checks if an episode ended ---
                done[i] = done[i] | terminated | truncated
                #^ ----------------------------------

        
        #~ --- saves the state for the next traj ---
        self.__curent_obs = obs
        self.__curent_info = info
        #~ -----------------------------------------
        
        mean_score = np.mean(score)
        return mean_score
    @override
    class TrainingStats(NetworkOptimiser.TrainingStats):
        """Data structure to hold various statistics logged during SAC training.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Attributes:
            score (float): Current score or metric used by the routine.
            q_losses (Tuple[np.ndarray[np.float32]]): Array containing q losses values.
            pi_loss (np.ndarray[np.float32]): Array containing pi loss values.
            alpha_loss (np.ndarray[np.float32]): Array containing alpha loss values.
        
        Example:
            >>> obj = TrainingStats()
        """



        q_losses:Tuple[np.ndarray[np.float32]]
        pi_loss:np.ndarray[np.float32]
        alpha:float
    @override
    @dataclass
    class HyperParams(NetworkOptimiser.HyperParams):
        """Data class for configuring SAC algorithm hyperparameters.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Attributes:
            epoch_count (int): Number of epoch used by the routine.
            initial_exploration_length (int): Value supplied for `initial_exploration_length`.
            rollout_size (int): Maximum number of transitions stored by the buffer.
            step_amount (int): Number of environment steps to collect during the call.
            batch_size (int): Number of samples to draw or process at once.
            gamma (float): Discount or exponential-decay factor used by the algorithm.
            log_alpha (np.ndarray): Array containing log alpha values.
            tau (float): Value supplied for `tau`.
            tarr_entropy (float): Value supplied for `tarr_entropy`.
            trainable_alpha (bool): Value supplied for `trainable_alpha`.
            normalize_obs (bool): Value supplied for `normalize_obs`.
            test_network_at_epoch (int): Value supplied for `test_network_at_epoch`.
            reward_scale (float): Value supplied for `reward_scale`.
        
        Example:
            >>> obj = HyperParams()
        """
        end_epoch: int = 100
        start_epoch: int = 1
        initial_exploration_length: int = 10_000
        rollout_size: int = 1_000_000
        step_amount: int = 256
        batch_size: int = 256
        gamma: float = 0.95
        log_alpha: np.ndarray = None #ikd whats a good default value
        tau: float = 0.005
        tarr_entropy: float = None
        trainable_alpha: bool = True
        normalize_obs: bool = True
        test_network_at_epoch: int = 5
        reward_scale: float = 1
    def setup_training(self,
                       hyper_params: HyperParams=None,
                       obs_normalizer: Normalizer=None,
                       schedulers: dict[str, scheduler.Scheduler]|None=None) -> None:
        """Prepares the SACOptimiser for training by initializing optimizers, normalizers, and internal state.
        
        Args:
            hyper_params (HyperParams): The hyperparameter setup. Defaults to standard SAC.
                                        defaults.
            obs_normalizer (Normalizer): Formatter for states. Instantiated if none provided.
            schedulers (dict[str, scheduler.Scheduler] | None): Optional learning rate.
                                                                schedulers. Defaults to None.
        """
        if schedulers is None:
            self.__schedulers = {}
        else:
            self.__schedulers = schedulers
        
        if hyper_params is None:
            hyper_params = self.HyperParams()
        
        if hyper_params.trainable_alpha:
            if hyper_params.tarr_entropy == None:
                hyper_params.tarr_entropy = - self.__action_dim
            if hyper_params.log_alpha == None:
                hyper_params.log_alpha = np.array(-1.6) # so alpha = 0.2
            self.__alpha_optimizer = optim.AdamW(alpha=3e-4,maximise=False)
            self.__alpha_optimizer.initialize(params=hyper_params.log_alpha)
        
        self.__hyper_params = hyper_params
        
        
        for i in range(len(self.__qs)):
            self.__q_optimizers[i].initialize(params=self.__qs[i].params,
                                              weight_mask=self.__qs[i].w_mask)
        
        self.__pi_optimizer.initialize(params=self.__pi.params,
                                       weight_mask=self.__pi.w_mask)
        
        if self.__rollout_buffer is not None:
            rollout_buffer = self.__rollout_buffer
        else:
            rollout_buffer = SACRolloutBuffer(obs_dim_count=self.__obs_dim,
                                              action_dim_count=self.__action_dim,
                                              rollout_size=hyper_params.rollout_size)
            self.__rollout_buffer = rollout_buffer
        
        if obs_normalizer is not None:
            self.__obs_normalizer = obs_normalizer
        elif hyper_params.normalize_obs:
            self.__obs_normalizer = RunningMeanStd(
                data_length=self.__env.single_observation_space.shape[0]
            )
        else:
            self.__obs_normalizer = NoNormalizer()
        self.__training_ready = True
    
    OptimiserCallback = Callable[[TrainingStats, "SACOptimiser"],bool]
    def train(self,
              callback: Optional[OptimiserCallback] = None) -> None:
        """Executes the SAC training loop, coordinating data collection, network updates, and logging."""
        if self.hyper_params.start_epoch > self.hyper_params.end_epoch:
            print_red_warning_simple("Tried start epoch > end epoch so no training happend")
            return
        hyper_params = self.__hyper_params
        
        #^ --- gets the first obs of the env ---
        self.__curent_obs, self.__curent_info = self.__env.reset(seed=int(rng.integers(0,100_000_000)))
        #^ -------------------------------------
        
        # --- stats variable creation ---
        score = float("-inf")
        training_starting_time = time.perf_counter()
        # -------------------------------
        
        #* --- initial random exploration ---
        if hyper_params.initial_exploration_length > 0:
            self.__get_steps(rollout_buffer=self.__rollout_buffer,
                                    obs_normalizer=self.__obs_normalizer,
                                    step_amount=hyper_params.initial_exploration_length,
                                    hyper_params=hyper_params,
                                    random_actions=True)
        #* ---------------------------------
        
        for i in range(hyper_params.start_epoch, hyper_params.end_epoch + 1):
            
            # --- time variable ---
            traj_start_time = time.perf_counter()
            # ---------------------
            
            #* --- gets new data ---
            self.__get_steps(rollout_buffer=self.__rollout_buffer,
                                    obs_normalizer=self.__obs_normalizer,
                                    hyper_params=hyper_params,
                                    step_amount=hyper_params.step_amount)
            #* ---------------------
            
            # --- time variable ---
            traj_dt = time.perf_counter() - traj_start_time
            params_update_start_time = time.perf_counter()
            # ---------------------
            
            #* --- train on the data ---
            q_losses, pi_loss, alpha_loss = self.__update_networks(rollout_buffer=self.__rollout_buffer,
                                                                   obs_normalizer=self.__obs_normalizer,
                                                                   hyper_params=hyper_params,
                                                                   step_amount=hyper_params.step_amount)
            #* -------------------------
            
            # --- time variable ---
            params_update_dt = time.perf_counter() - params_update_start_time
            # ---------------------
            
            # #* --- test the network ---
            if i % hyper_params.test_network_at_epoch == 0:
                score = self.test_pi(obs_normalizer=self.__obs_normalizer,
                                     run_amount=10)
            #* ------------------------
            
            for sched in self.__schedulers:
                self.__schedulers[sched].step(score=score)
            
            
            # --- stats update ---
            total_training_time = time.perf_counter() - training_starting_time
            # --------------------
            
            #^ --- calling the callback ---
            if callback is not None:
                stat_dict : SACOptimiser.TrainingStats = {
                    "current_epoch":i,
                    "number_of_steps":self.__rollout_buffer.index,
                    "total_param_update":self.__n_of_params_update,
                    "params_updating_dt":params_update_dt,
                    "trajectories_collecting_dt":traj_dt,
                    "total_training_time":total_training_time,
                    "alpha":float(np.exp(self.__hyper_params.log_alpha)),
                    "pi_loss":pi_loss,
                    "q_losses":q_losses,
                    "score":score,
                }
                training_ended = callback(stat_dict, self)
                if training_ended:
                    print("training stoped by callback")
                    break
            #^ ----------------------------
    @override
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through the actor network.
        
        Args:
            obs (np.ndarray): Observation array used by the routine.
        
        Returns:
            np.ndarray: Computed result expressed as `np.ndarray`.
        
        Example:
            >>> obj = SACOptimiser(...)
            >>> result = obj.forward(obs=...)
        """
        if not self.__training_ready:
            raise nn.InitializationException("you need to call setup_training before calling forward")
        if self.__obs_normalizer is not None:
            obs = self.__obs_normalizer.normalize(obs)
        return self.__pi.deterministic_forward(obs)
    @property
    def env(self):
        """Returns the current environment instance used for training.
        
        Returns:
            Any: The current environment instance.
        """
        return self.__env
    @property
    def alpha_optimizer(self):
        """Returns the current optimizer instance for the entropy temperature coefficient (alpha).
        
        Returns:
            Any: The optimizer used for updating alpha.
        """
        return self.__alpha_optimizer
    @property
    def rollout_buffer(self):
        """Returns the current rollout buffer instance used for experience storage.
        
        Returns:
            Any: The current SACRolloutBuffer instance.
        """
        return self.__rollout_buffer
    @property
    def hyper_params(self):
        """Returns the current hyperparameters used for training.
        
        Returns:
            Any: The current hyperparameter configuration object.
        """
        return self.__hyper_params
    @property
    def pi(self):
        """Returns the current policy network.
        
        Returns:
            Any: The current actor network instance.
        """
        return self.__pi
    @property
    def qs(self):
        """Returns the current Q-networks.
        
        Returns:
            Any: A list containing the current Q-network instances.
        """
        return self.__qs
    @property
    def q_tarrs(self):
        """Returns the current target Q-networks.
        
        Returns:
            Any: A list containing the current target Q-network instances.
        """
        return self.__q_tarrs
    @property
    def alpha(self):
        """Returns the current value of the entropy temperature coefficient (alpha).
        
        Returns:
            Any: The current alpha value.
        """
        return np.exp(self.__hyper_params.log_alpha)
    @property
    def schedulers(self):
        """Returns the dictionary of learning rate schedulers.
        
        Returns:
            Any: The current schedulers in use.
        """
        return self.__schedulers
    @property
    def pi_optimizer(self):
        """Returns the policy optimizer instance.
        
        Returns:
            Any: The optimizer used for updating the policy network.
        """
        return self.__pi_optimizer
    @property
    def q_optimizers(self):
        """Returns the list of Q-network optimizer instances.
        
        Returns:
            Any: A list of optimizers used for updating the Q-networks.
        """
        return self.__q_optimizers
    @property
    def obs_normalizer(self):
        """Returns the observation normalizer used by the current training run."""
        return self.__obs_normalizer
