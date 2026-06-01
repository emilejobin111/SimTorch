"""Shared reinforcement-learning optimizer interfaces.

This module defines the abstract training contract, callback signature, and
training-statistics schema consumed by the concrete RL optimizers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypedDict
import numpy as np


class NetworkOptimiser(ABC):
    """Optimizer implementing the network training procedure.
    
    The implementation coordinates runtime state, optimization steps, and
    checkpoint-friendly serialization helpers needed by the training workflow.
    """
    class HyperParams:
        """Base marker class for algorithm-specific hyperparameter containers.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Example:
            >>> obj = HyperParams()
        """
        ...
    class TrainingStats(TypedDict):
        """Typed dictionary describing the statistics reported during training.
        
        Concrete algorithms subclass or extend this container to describe their own
        configuration or reporting payloads.
        
        Attributes:
            trajectories_collecting_dt (float): Value supplied for `trajectories_collecting_dt`.
            params_updating_dt (float): Value supplied for `params_updating_dt`.
            total_training_time (float): Value supplied for `total_training_time`.
            number_of_steps (int): Value supplied for `number_of_steps`.
            current_epoch (int): Value supplied for `current_epoch`.
            total_param_update (int): Value supplied for `total_param_update`.
        
        Example:
            >>> obj = TrainingStats()
        """
        score:float
        trajectories_collecting_dt:float
        params_updating_dt:float
        total_training_time:float
        number_of_steps: int
        current_epoch: int
        total_param_update: int
    
    def setup_training(self,
                       hyper_params: HyperParams,
                       obs_normalizer: Any,
                       schedulers: Optional[Any] = None) -> None:
        """Configure the optimizer for training with the provided components.
        
        args:
            hyper_params (HyperParams): Algorithm-specific hyperparameters.
            obs_normalizer (Any): Normalizer for environment observations.
            schedulers (Optional[Any]): Optional learning-rate schedulers for training.
        """
        ...
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Performs a forward pass using the main (unmutated) neural network.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        output of the main network. The method is used during environment interaction and may
        be called multiple times per training step.
        
        Args:
            obs (np.ndarray): Current environment observations.
        
        Returns:
            np.ndarray: Output of the main network, typically representing action logits or values.
        
        Example:
            >>> obj = NetworkOptimiser(...)
            >>> output = obj.forward(obs=np.array(...))
        """
        ...
    OptimiserCallback = Callable[[TrainingStats, "NetworkOptimiser"],bool]
    
    @abstractmethod
    def train(self,
              callback: Optional[OptimiserCallback] = None) -> None:
        """Run the training loop for the configured workflow.
        
        The routine coordinates environment interaction, gradient updates, callbacks, and
        auxiliary bookkeeping required by the algorithm.
        
        Args:
            callback (Optional[OptimiserCallback]): Callback invoked with training statistics.
                                                    during execution.
        
        Returns:
            None: This method performs training side effects and returns no value.
        
        Example:
            >>> obj = NetworkOptimiser(...)
            >>> result = obj.train(hyper_params=..., obs_normalizer=..., schedulers=..., ...)
        """
        ...
