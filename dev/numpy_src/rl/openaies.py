"""OpenAI Evolution Strategies training utilities.

This module implements an evolutionary reinforcement-learning optimizer that evaluates
mirrored parameter perturbations and updates a policy network from aggregate episode
scores.
"""


from __future__ import annotations
from typing import override, Tuple, List,  Any, Callable, Optional, TypedDict
from dataclasses import dataclass
import os
from abc import ABC, abstractmethod
from copy import deepcopy
import time
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from ..core import nn
from ..core import optim
from ..core import scheduler
from ..core.rng import rng


from .utils import normalize_values,Normalizer, NoNormalizer, RunningMeanStd, print_red_warning_simple
from .base_nn_optimiser import NetworkOptimiser




class EnvException(Exception):
    """Custom exception raised when the environment is incompatible with the neural network."""
    ...




class OpenAIES(NetworkOptimiser):
    """Implementation of the OpenAI Evolution Strategies (ES) algorithm.
    
    This class handles the evaluation and derivative-free optimization of a neural network
    acting within a Gymnasium environment.
    
    Args:
        env (gym.Env): The Gymnasium environment to solve.
        network (nn.Sequential): The neural network model to optimize.
        optimizer (optim.GradientOptimizer): The optimizer used to apply gradient updates.
    
    Example:
        >>> obj = OpenAIES(env=env, network=..., optimizer=...)
    """
    def __init__(self,env:gym.Env,network:nn.Sequential,optimizer:optim.GradientOptimizer):
        """Initialize a new OpenAIES instance.
        
        Args:
            env (gym.Env): Gymnasium environment used for interaction or validation.
            network (nn.Sequential): Neural network instance used by the routine.
            optimizer (optim.GradientOptimizer): Gradient optimizer responsible for parameter.
                                                 updates.
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            EnvException: Raised if the environment is incompatible with the algorithm.
        """
        self.__env = env
        self.__main_network = network
        self.__optimizer = optimizer
        
        self.__is_box = isinstance(self.__env.action_space, gym.spaces.Box)
        self.__is_discrete = isinstance(self.__env.action_space, gym.spaces.Discrete)
        if self.__is_box:
            self.__low, self.__high = self.__env.action_space.low, self.__env.action_space.high
        
        
        
        self.__total_steps = 0
        
        
        if not isinstance(network,nn.Sequential):
            raise TypeError("OpenAIES only suport nn.Sequential as an optimisable network")
        if not self.validate_env(): 
            raise EnvException("The environnement is not compatible with the neural network chosen")
        
        self.__mutant_network = deepcopy(network)
        self.__training_ready = False
    

    def validate_env(self) -> bool:
        """Tests if the neural network architecture is compatible with the environment's input and output spaces.
        
        The validation checks guard algorithm assumptions early so training fails fast when the
        environment shape or action-space type is incompatible.
        
        Returns:
            bool: True if the forward pass succeeds and output shape matches the action space,
                  False otherwise.
        
        Example:
            >>> obj = OpenAIES(...)
            >>> result = obj.validate_env()
        """
        
        # I honnestly didn't do this function, Gemini did.
        # the size and shape thing is soo confusing
        
        observation = self.__env.observation_space.sample()
        try:
            action_predite = self.__main_network.forward(observation)
        except Exception:
            return False
        
        action_space = self.__env.action_space
        # if the action space is discrete
        if isinstance(action_space, gym.spaces.Discrete):
            return action_predite.shape[-1] == action_space.n
        
        # if the action space is continuous
        elif isinstance(action_space, gym.spaces.Box):
            return action_predite.shape[-len(action_space.shape):] == action_space.shape
            
        else:
            return False
        
    
    def evaluate_main(self, obs_normalizer: Normalizer,num_episode:int=10,) -> float:
        """Evaluates the performance of the current main network without any mutations.
        
        Args:
            obs_normalizer (Normalizer): Observation normalizer applied before network.
                                         evaluation.
            num_episode (int): The number of episodes to run for the evaluation. Defaults to 10.
        
        Returns:
            float: The average score achieved over the specified number of episodes.
        
        Example:
            >>> obj = OpenAIES(...)
            >>> result = obj.evaluate_main(obs_normalizer=..., num_episode=4)
        """
        total_score = 0.0
        for _ in range(num_episode):
            # so that it does +0 to all params of the main
            # so it evluate the raw params of the main
            total_score += self.__evaluate_epsilon(hyper_params=self.HyperParams(),
                                                   epsilon=0,
                                                   obs_normalizer=obs_normalizer,
                                                   update_normalizer=False) 
        
        #? takes the avarages score
        return total_score/num_episode

    def __evaluate_epsilon(self,
                           hyper_params: HyperParams, 
                           epsilon: np.ndarray | int,
                           obs_normalizer: Normalizer, 
                           test_seed: int | None = None,
                           update_normalizer: bool=True) -> float:
        """Evaluates a mutated version of the network in the environment for a single episode.
        
        Applies the given epsilon (noise) scaled by sigma to the main network's parameters,
        interacts with the environment until completion, and returns the accumulated score.
        
        Args:
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
            epsilon (np.ndarray | int): The noise vector to add to the parameters. If 0,.
                                        evaluates the unmodified main network.
            obs_normalizer (Normalizer): Observation normalizer applied before network.
                                         evaluation.
            test_seed (int | None): The seed for the environment to ensure reproducibility.
                                    across mutants. Defaults to None.
            update_normalizer (bool): Normalizer responsible for maintaining or applying.
                                      observation statistics.
        
        Returns:
            float: The total reward (score) achieved by the mutated network in one episode.
        
        Example:
            >>> obj = OpenAIES(...)
            >>> result = obj.__evaluate_epsilon(hyper_params=..., epsilon=0.1, obs_normalizer=..., ...)
        """
        
        #? --- making so the espilon is evaluated ---
        self.__mutant_network.params_copy(self.__main_network.params + epsilon * hyper_params.sigma)
        #? ------------------------------------------
        
        #^ --- reseting the state with the right env seed ---
        obs:np.ndarray
        obs, info = self.__env.reset(seed=test_seed)
        #^ --------------------------------------------------
        
        #~ --- obs normalisation ---
        if update_normalizer:
            obs = obs_normalizer.normalize_and_update(obs.reshape(1,-1))
        else : 
            obs = obs_normalizer.normalize(obs.reshape(1,-1))
        #~ -------------------------
        
        # --- for graphing ---
        self.__total_steps += 1
        # --------------------
        
        
        #^ --- Setting up the vars ---
        score = 0.0
        done = False
        #^ ---------------------------
        
        
        #? --- Evaluation loop ---
        while not done:
            raw_action = self.__mutant_network.forward(obs)
            
            # --- flatten the array ---
            raw_action = raw_action.ravel()
            # -------------------------
            
            #& --- choose the right action type depending on the action space type ---
            if self.__is_box:
                action = np.clip(raw_action, self.__low, self.__high)
                
                
            elif self.__is_discrete:
                action = int(np.argmax(raw_action))
            else:
                action = raw_action
            #& -----------------------------------------------------------------------
            
            #^ --- Environnement step ---
            obs, reward, terminated, truncated, info = self.__env.step(action)
            #^ --------------------------
            
            
            #~ --- obs normalisation ---
            if update_normalizer:
                obs = obs_normalizer.normalize_and_update(obs.reshape(1,-1))
            else : 
                obs = obs_normalizer.normalize(obs.reshape(1,-1))
            #~ -------------------------
            
            #^ --- adding the score to the total score ---
            score += float(reward)
            #^ -------------------------------------------
            
            #^ --- To know if the episode is finnished ---
            done = terminated or truncated
            #^ -------------------------------------------
            
            # --- for graphing ---
            self.__total_steps += 1
            # --------------------
        #? ----------------------
        
        return score
    
    def __generate_epsillons(self, hyper_params: HyperParams) -> Tuple[np.ndarray,int]:
        """Generates the noise vectors (epsilons) used to mutate the network's parameters.
        
        Args:
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
        
        Returns:
            Tuple[np.ndarray, int]: A tuple containing the matrix of generated epsilons and the
                                    actual number of mutants (which may be adjusted if mirrored
                                    is True and num_mutant was odd).
        
        Example:
            >>> obj = OpenAIES(...)
            >>> result = obj.__generate_epsillons(hyper_params=...)
        """
        #? --- to make sure the distr is centered ---
        if hyper_params.mirrored:
            
            #& --- to actually generate n amount of mutants ---
            half_pop = hyper_params.num_mutant // 2
            #& ------------------------------------------------
            
            #! catch if num_mutant == 1 so num//2 = 0 and thats not good
            if hyper_params.num_mutant == 1:
                half_pop = 1
            
            #^ --- generate the epsillons (mu=0 and std = 1) ---
            epsillons = rng.normal(size=(half_pop,len(self.__main_network.params))).astype(dtype=np.float32)  
            #^ -------------------------------------------------
            
            #^ --- creates the epsillons copy ---
            epsillons = np.vstack((epsillons,-epsillons))
            #^ ----------------------------------
            
            #& --- to ajust if num_mutant was odd ---
            hyper_params.num_mutant = len(epsillons)
            #& --------------------------------------
        #? ------------------------------------------
        
        else:
            #^ --- generate the epsillons (mu=0 and std = 1) ---
            epsillons = rng.normal(size=(hyper_params.num_mutant,len(self.__main_network.params))).astype(dtype=np.float32)  
            #^ -------------------------------------------------
        
        return epsillons
    
    
    def __approximate_gradient(self,
                               hyper_params: HyperParams,
                               obs_normalizer:Normalizer) -> Tuple[np.ndarray,np.ndarray]:
        """Approximates the gradient of the expected reward with respect to the network parameters.
        
        Uses the Evolution Strategies method to estimate the gradient by evaluating a population
        of mutated networks.
        
        Args:
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
            obs_normalizer (Normalizer): Observation normalizer applied before network.
                                         evaluation.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The approximated gradient vector.
        
        Example:
            >>> obj = OpenAIES(...)
            >>> result = obj.__approximate_gradient(hyper_params=..., obs_normalizer=...)
        """
        
        
        epsillons = self.__generate_epsillons(hyper_params=hyper_params)
        
        
        #^ --- all the scores for all mutant starts to zero ---
        total_scores = np.zeros((hyper_params.num_mutant,), dtype=np.float32)
        #^ ----------------------------------------------------
        
        #? --- main evaluating loop ---
        for _ in range(hyper_params.num_episodes):
            
            #& --- to make sure that all the mutants are tested on the same random env ---
            seed = int(rng.integers(0, 10000000))
            #& ---------------------------------------------------------------------------
            
            #^ --- evaluating all mutant one by one for this episodes ---
            for i in range(hyper_params.num_mutant):
                epsillon = epsillons[i]
                total_scores[i] += self.__evaluate_epsilon(hyper_params=hyper_params,
                                                           epsilon=epsillon,
                                                           obs_normalizer=obs_normalizer,
                                                           test_seed=seed)
            #^ -----------------------------------------------------------
        #? -------------------------
        
        
        #? --- avarage scores across all episodes ---
        avarage_scores = total_scores/hyper_params.num_episodes
        #? ------------------------------------------
        
        
        #^ --- to make sure the std = 1 and mean = 0 ---
        scores_norm = normalize_values(avarage_scores)
        #^ ---------------------------------------------
        
        #^ --- the math formula to approx the gradient -------
        weighted_epsilons = scores_norm.reshape(-1,1)*epsillons
        gradient_approx = np.sum(weighted_epsilons,axis=0) / (hyper_params.num_mutant * hyper_params.sigma)
        #^ ---------------------------------------------------
        
        return gradient_approx, avarage_scores
    @override
    class TrainingStats(NetworkOptimiser.TrainingStats):
        """Typed dictionary describing the statistics reported during training.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Attributes:
            score (float): Value supplied for `score`.
            epsilon_scores (np.ndarray): Array containing epsilon scores values.
        
        Example:
            >>> obj = TrainingStats()
        """
        epsilon_scores:np.ndarray
    @dataclass
    class HyperParams(NetworkOptimiser.HyperParams):
        """Configuration container for OpenAI Evolution Strategies hyperparameters.
        
        This dataclass defines the hyperparameters required to configure and run the 
        ES optimization loop. It controls the population size, evaluation length, 
        noise scale, and overall training duration.
        
        Attributes:
            num_mutant (int): The number of perturbed networks (mutants) to generate 
                and evaluate per epoch. Defaults to 50.
            num_episodes (int): The number of environment episodes to run per mutant 
                to estimate its fitness score. Defaults to 1.
            start_epoch (int): The starting epoch index for the training loop. Defaults to 1.
            end_epoch (int): The final epoch index for the training loop. Defaults to 100.
            sigma (float): The standard deviation (scale) of the Gaussian noise applied 
                to the main network's parameters. Defaults to 0.05.
            mirrored (bool): Whether to use mirrored (antithetic) sampling by evaluating 
                both positive and negative perturbations for variance reduction. 
                Defaults to False.
            normalize_obs (bool): Whether to automatically initialize a running 
                observation normalizer if none is explicitly provided. Defaults to True.
        
        Example:
            >>> params = OpenAIES.HyperParams(num_mutant=100, sigma=0.1, mirrored=True)
        """
        num_mutant:int=50
        num_episodes: int=1
        start_epoch: int=1
        end_epoch: int=100
        sigma: float=0.05
        mirrored: bool = False
        normalize_obs: bool = True
        
    def setup_training(self, hyper_params: HyperParams, obs_normalizer: Normalizer=None, schedulers: dict[str,scheduler.Scheduler]=None):
        """Sets up the training configuration for the optimizer.
        
        This method initializes the necessary components and configurations before starting the
        training loop. It can be used to prepare the observation normalizer, schedulers, and any
        other stateful components required for training.
        
        Args:
            hyper_params (HyperParams): Hyperparameter object controlling the training loop.
            obs_normalizer (Normalizer): Observation normalizer applied before network.
                                         evaluation.
            schedulers (dict[str, scheduler.Scheduler]): Named schedulers updated alongside the training loop.
        """
        #? --- gives the right params to optimize to the optimizer ---
        self.__optimizer.initialize(self.__main_network.params)
        #? -----------------------------------------------------------
        
        # --- seting up the schedulers ---
        if schedulers is None:
            self.__schedulers = []
        else:
            self.__schedulers = schedulers
        # --------------------------------
        if hyper_params is None:
            hyper_params = self.HyperParams()
        self.__hyper_params = hyper_params
        
        if obs_normalizer is not None:
            self.__obs_normalizer = obs_normalizer
        elif hyper_params.normalize_obs:
            self.__obs_normalizer = RunningMeanStd(
                data_length=self.__env.observation_space.shape[0]
            )
        else:
            self.__obs_normalizer = NoNormalizer()
        self.__training_ready = True
    
    OptimiserCallback = Callable[[TrainingStats, "OpenAIES"],bool]
    def train(self,
              callback: Optional[OptimiserCallback] = None
              ) -> None:
        """Trains the main neural network using the Evolution Strategies algorithm.
        
        This method iteratively approximates the gradient and applies updates via the optimizer.
        All training hyperparameters are injected directly here to control the population size,
        episodes, noise scaling, and early stopping criteria.
        
        Args:
            callback (Optional[OptimiserCallback]): Callback invoked with training statistics.
                                                    during execution.
        
        Returns:
            None: This method performs training side effects and returns no value.
        
        Example:
            >>> obj = OpenAIES(...)
            >>> obj.setup_training(hyper_params=..., obs_normalizer=..., schedulers=...)
            >>> obj.train(callback=...)
            >>> # testing it
            >>> obj.forward(obs=...)
        """
        if not self.__training_ready:
            raise Exception("You must call setup_training before calling train")
        if not isinstance(callback, Callable) and callback is not None:
            raise TypeError("Callback must be a callable function that takes in training stats and the optimiser instance, and returns a boolean.")
        if self.hyper_params.start_epoch > self.hyper_params.end_epoch:
            print_red_warning_simple("Tried start epoch > end epoch so no training happend")
            return
        
        # --- stats variable creation ---
        training_starting_time = time.perf_counter()
        # -------------------------------
        
        #? --- main training loop ---
        for epoch in range(self.__hyper_params.start_epoch, self.__hyper_params.end_epoch + 1):
            
            # --- time variable ---
            traj_start_time = time.perf_counter()
            # ---------------------
            
            #^ --- approximates the gradient so that the optimizer can do it's job ---
            gradient, scores = self.__approximate_gradient(self.__hyper_params, self.__obs_normalizer)
            #^ -----------------------------------------------------------------------
            
            # --- time variable ---
            traj_dt = time.perf_counter() - traj_start_time
            params_update_start_time = time.perf_counter()
            # ---------------------
            
            #^ --- the optimizer optimize the function from the gradient ---
            self.__optimizer.step(gradient)
            #^ -------------------------------------------------------------
            
            # --- time variable ---
            params_update_dt = time.perf_counter() - params_update_start_time
            # ---------------------
            
            #? --- to update the user about the training process ---
            score = self.evaluate_main(num_episode=10,obs_normalizer=self.__obs_normalizer)
            #? -----------------------------------------------------
            
            #? --- Schedulers update ---
            for sched in self.__schedulers:
                self.__schedulers[sched].step(score)
            #? -------------------------
            
            
            # --- stats update ---
            total_training_time = time.perf_counter() - training_starting_time
            # --------------------
            
            #^ --- calling the callback ---
            if callback is not None:
                stat_dict: OpenAIES.TrainingStats = {
                    "current_epoch" : epoch,
                    "number_of_steps" : self.__total_steps,
                    "total_param_update" : epoch,
                    "params_updating_dt":params_update_dt,
                    "trajectories_collecting_dt":traj_dt,
                    "total_training_time":total_training_time,
                    "score":score,
                    "epsilon_scores":scores
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
            >>> obj = PPOptimiser(...)
            >>> result = obj.forward(obs=...)
        """
        if not self.__training_ready:
            raise nn.InitializationException("you need to call setup_training before calling forward")
        if self.__obs_normalizer is not None:
            obs = self.__obs_normalizer.normalize(obs)
        return self.__main_network.forward(obs)
    @property
    def env(self) -> gym.Env:
        return self.__env
    @property
    def main_network(self):
        """Return the current main network.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__main_network
    @property
    def optimizer(self):
        """Return the current optimizer.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__optimizer
    @property
    def hyper_params(self):
        """Return the current hyper parameters."""
        return self.__hyper_params
    @property
    def obs_normalizer(self):
        """Return the observation normalizer used by the current training run."""
        return self.__obs_normalizer
    @property
    def schedulers(self):
        """Return the schedulers used by the current training run."""
        return self.__schedulers