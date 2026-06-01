"""Worker abstractions for launching SimTorch reinforcement-learning jobs.

This module adapts the concrete optimizer classes into higher-level worker objects that
separate environment setup, runtime state, and training calls.
"""

# --- base python import ---
from __future__ import annotations
from typing import override, Tuple, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from copy import deepcopy
from ast import literal_eval
from pathlib import Path
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


# --- import to save ---
from ..save.checkpoint import SACState, PPOState, OpenAIESState, CheckPoint, OPTIM_FILE_EXTENSION
from ..save.utils import save_checkpoint_from_openaies, save_checkpoint_from_ppo, save_checkpoint_from_sac, load_checkpoint_to_openaies, load_checkpoint_to_ppo, load_checkpoint_to_sac
# ----------------------

# --- SimTorch.rl imports ---
from ..rl import policy
from ..rl.utils import Normalizer
from ..rl.base_nn_optimiser import NetworkOptimiser
from ..rl.openaies import OpenAIES
from ..rl.sac_optimiser import SACOptimiser
from ..rl.ppoptimiser import PPOptimiser
# -----------------------------

class OptimType:
    PPO = "PPOtimiser"
    SAC = "SACOptimiser"
    OpAIES = "OpenAIES"

class TrainingStatus(Enum):
    untrained = 0
    training = 1
    trained = 2

class TrainingInfo(TypedDict):
    """Data structure representing the information of a saved training session."""
    name: str
    training_status: int  # Enum value (int) pour être sérialisable en JSON
    path: str
    algo: str             # OptimType
    score: float


class RLWorker(ABC):
    """Worker wrapper that configures and launches the rl training flow."""

    def __init__(self,
                 worker_callback: Optional[NetworkOptimiser.OptimiserCallback] = None,
                 checkpoint_path: Optional[str | Path] = None):
        
        self._worker_callback = worker_callback
        
        if checkpoint_path:
            self._checkpoint_path = Path(checkpoint_path)
            name = self._checkpoint_path.stem
            path_str = str(self._checkpoint_path)
        else:
            self._checkpoint_path = None
            name = ""
            path_str = ""

        # Initialisation du dictionnaire d'informations
        self._training_info: TrainingInfo = {
            "name": name,
            "training_status": TrainingStatus.untrained.value,
            "path": path_str,
            "algo": "",
            "score": float('-inf')
        }

        self._is_training = False
        self._is_initialized = False 
        self._env: Optional[gym.Env | gym.vector.VectorEnv] = None

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: Optional[str | Path]):
        if value:
            self._checkpoint_path = Path(value)
            self._training_info["name"] = self._checkpoint_path.stem
            self._training_info["path"] = str(self._checkpoint_path)
        else:
            self._checkpoint_path = None
            self._training_info["name"] = ""
            self._training_info["path"] = ""

    @property
    def is_training(self) -> bool:
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"is_training must be a boolean, got {type(value).__name__}")
        self._is_training = value

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @is_initialized.setter
    def is_initialized(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"is_initialized must be a boolean, got {type(value).__name__}")
        self._is_initialized = value
    
    @property
    def env(self):
        return self._env

    @property
    def worker_callback(self):
        return self._worker_callback

    @abstractmethod
    def setup_from_checkpoint(self, num_env: Optional[int] = 1):
        """Setup the worker from its assigned checkpoint file."""
        ...
    
    @abstractmethod
    def save_checkpoint(self, optim: NetworkOptimiser, epoch: int=1):
        """Save a checkpoint of the current training state."""
        ...
    
    def setup_env(self, env: gym.Env | gym.vector.VectorEnv):
        self._env = env
    
    def optimiser_callback(self, training_stats: NetworkOptimiser.TrainingStats, nn_optimiser: NetworkOptimiser) -> bool:
        """Handle optimiser callback and save TrainingInfo on score improvement, end of training, or pause."""
        
        current_score = training_stats.get("score")
        current_epoch = training_stats.get("current_epoch", 1)

        # 1. Update best score
        if current_score is not None and current_score > self._training_info["score"]:
            self._training_info["score"] = float(current_score)
            # Save current_epoch + 1 to resume from the next epoch in case of a crash
            self.save_checkpoint(nn_optimiser, epoch=current_epoch + 1)

        # 2. Check for natural end of training
        if current_epoch >= nn_optimiser.hyper_params.end_epoch:
            self.is_training = False
            self._training_info["training_status"] = TrainingStatus.trained.value
            # Save checkpoint to finalize the end of training
            self.save_checkpoint(nn_optimiser, epoch=current_epoch + 1)
            return True  # Force training loop termination

        # 3. Check UI commands (Pause / Stop)
        should_stop = False
        if self._worker_callback is not None:
            should_stop = self._worker_callback(training_stats, nn_optimiser)

        if should_stop:
            # Save exact progress at the moment of pause
            self.save_checkpoint(nn_optimiser, epoch=current_epoch + 1)

        return should_stop
    
    @abstractmethod
    def setup(self, *args, **kwargs): 
        pass
    
    @abstractmethod
    def train(self):
        pass

###########################################################################

class OpenAIESWorker(RLWorker):
    def __init__(self, worker_callback = None, checkpoint_path: Optional[str | Path] = None):
        self.__can_save = False
        super().__init__(worker_callback, checkpoint_path)
        self._training_info["algo"] = OptimType.OpAIES

    @override
    def setup_from_checkpoint(self, num_env: Optional[int] = 1):
        if self._checkpoint_path is None:
            raise ValueError("A checkpoint_path must be provided to load a checkpoint.")
            
        optim, metadata = load_checkpoint_to_openaies(checkpoint=self._checkpoint_path, num_envs=num_env)
        
        # On charge l'ancien TrainingInfo s'il y en a un
        if metadata is not None:
            self._training_info.update(metadata)
            
        self.is_initialized = True
        self.__optim = optim
        self.__can_save = True

    @override
    def save_checkpoint(self, optim: OpenAIES, epoch: int=1):
        if not self.__can_save or self._checkpoint_path is None:
            print("Checkpoint has been trying to be saved, but saving is disabled for this worker (no path provided).")
            return
        # Sauvegarde directe de self._training_info comme meta_data
        save_checkpoint_from_openaies(optimiser=optim, path=self._checkpoint_path, step=epoch, meta_data=self._training_info)
    
    @override
    def setup(self, env: gym.Env, hyper_params:OpenAIES.HyperParams, network:nn.Sequential,
              optimizer:optim.GradientOptimizer, obs_normalizer: Optional[Normalizer] = None,
              schedulers: dict = None):
        self.setup_env(env)
        self.__optim = OpenAIES(env=self.env, network=network, optimizer=optimizer)
        self.__optim.setup_training(hyper_params=hyper_params, obs_normalizer=obs_normalizer, schedulers=schedulers)
        self.is_initialized = True
        
    @override
    def train(self):
        if not self.is_initialized:
            raise Exception("You must call setup() before train()")
            
        self._training_info["training_status"] = TrainingStatus.training.value
        self.__optim.train(callback=self.optimiser_callback)
        
        

class PPOWorker(RLWorker):
    def __init__(self, worker_callback = None, checkpoint_path: Optional[str | Path] = None):
        self.__can_save = False
        super().__init__(worker_callback, checkpoint_path)
        self._training_info["algo"] = OptimType.PPO

    @override
    def setup_from_checkpoint(self, num_env: Optional[int] = 1):
        if self._checkpoint_path is None:
            raise ValueError("A checkpoint_path must be provided to load a checkpoint.")
            
        optim, metadata = load_checkpoint_to_ppo(checkpoint=self._checkpoint_path, num_envs=num_env)
        
        # On charge l'ancien TrainingInfo s'il y en a un
        if metadata is not None:
            self._training_info.update(metadata)
            
        self.is_initialized = True
        self.__optim = optim
        self.__can_save = True

    @override
    def save_checkpoint(self, optim: PPOptimiser, epoch: int=1):
        if not self.__can_save or self._checkpoint_path is None:
            print("Checkpoint has been trying to be saved, but saving is disabled for this worker (no path provided).")
            return
        save_checkpoint_from_ppo(optimiser=optim, path=self._checkpoint_path, step=epoch, meta_data=self._training_info)
    
    @override
    def setup(self, env: gym.vector.VectorEnv, hyper_params:PPOptimiser.HyperParams,
              actor_net:policy.ActorNN, critic_net:policy.CriticNN,
              actor_optimizer:optim.GradientOptimizer, critic_optimizer:optim.GradientOptimizer,
              obs_normalizer: Optional[Normalizer] = None, schedulers: dict = None):
        self.setup_env(env)
        self.__optim = PPOptimiser(env=self.env, networks=(actor_net, critic_net),
                                   actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)
        self.__optim.setup_training(hyper_params=hyper_params, obs_normalizer=obs_normalizer, schedulers=schedulers)
        self.is_initialized = True
    
    @override
    def train(self):
        if not self.is_initialized:
            raise Exception("You must call setup() before train()")
            
        self._training_info["training_status"] = TrainingStatus.training.value
        self.__optim.train(callback=self.optimiser_callback)
        
        
class SACWorker(RLWorker):
    def __init__(self, worker_callback = None, checkpoint_path: Optional[str | Path] = None):
        self.__can_save = False
        super().__init__(worker_callback, checkpoint_path)
        self._training_info["algo"] = OptimType.SAC

    @override
    def setup_from_checkpoint(self, num_env: Optional[int] = 1):
        if self._checkpoint_path is None:
            raise ValueError("A checkpoint_path must be provided to load a checkpoint.")
            
        optim, metadata = load_checkpoint_to_sac(checkpoint=self._checkpoint_path, num_envs=num_env)
        
        # On charge l'ancien TrainingInfo s'il y en a un
        if metadata is not None:
            self._training_info.update(metadata)
            
        self.is_initialized = True
        self.__optim = optim
        self.__can_save = True
        
    @override
    def save_checkpoint(self, optim: SACOptimiser, epoch: int=1):
        if not self.__can_save or self._checkpoint_path is None:
            print("Checkpoint has been trying to be saved, but saving is disabled for this worker (no path provided).")
            return
        save_checkpoint_from_sac(optimiser=optim, path=self._checkpoint_path, step=epoch, meta_data=self._training_info)

    @override
    def setup(self, env: gym.vector.VectorEnv, hyper_params:PPOptimiser.HyperParams,
              actor_net:policy.ActorNN, q_networks:Tuple[policy.QNN, policy.QNN],
              actor_optimizer:optim.GradientOptimizer, q_optimizers:Tuple[optim.GradientOptimizer, optim.GradientOptimizer],
              obs_normalizer: Optional[Normalizer] = None, schedulers: dict = None):
        self.setup_env(env)
        self.__optim = SACOptimiser(env=self.env, actor_network=actor_net, q_networks=q_networks,
                                    actor_optimizer=actor_optimizer, q_optimizers=q_optimizers)
        self.__optim.setup_training(hyper_params=hyper_params, obs_normalizer=obs_normalizer, schedulers=schedulers)
        self.is_initialized = True
    
    @override
    def train(self):
        if not self.is_initialized:
            raise Exception("You must call setup() before train()")
            
        self._training_info["training_status"] = TrainingStatus.training.value
        self.__optim.train(callback=self.optimiser_callback)
        