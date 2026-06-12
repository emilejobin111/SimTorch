from __future__ import annotations
from typing import Tuple, override, Optional
from copy import deepcopy

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from ..core import nn






LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
        

class ActorNN(nn.Sequential):
    def __init__(self, *modules:nn.JaxModule, standalone_std:bool):
        super().__init__(*modules)
        self._standalone_std = standalone_std
    
    @classmethod
    def new(cls, *modules:nn.JaxModule, standalone_std:bool):
        modules:list[nn.JaxModule] = list(modules)
        if standalone_std:
            if not isinstance(modules[-1], nn.StdLayer):
                std_layer = nn.StdLayer.new(nn.Sequential.get_output_count(modules=modules),)
                modules.append(std_layer)
                
        else:
            for i in reversed(range(len(modules))):
                if isinstance(modules[i], nn.ActivationFunction):
                    pass
                elif isinstance(modules[i], nn.Linear):
                    old_mod = modules[i]
                    modules[i] = nn.Linear.new(modules[i].input_count,modules[i].output_count,modules[i]._use_bias)
                    del old_mod
                    
                    break
                elif isinstance(modules[i], nn.StdLayer):
                    raise nn.InitializationException("A ActorNN without standalone_std shouldn't posess an StdLayer in it's modules")
                else:
                    raise NotImplementedError(f"The module type {type(modules[i])} is not supported for adding a std output")
            raise nn.InitializationException("No module found in the actor network to add the std output to")
        
        return cls(*modules, standalone_std=standalone_std)
    
    def forward(self, input) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._standalone_std:
            return super().forward(input)
        else:
            output_c = self.get_output_count(modules=self._modules)//2
            
            act_log_std = super().forward(input=input)
            act = act_log_std[0:output_c]
            log_std = act_log_std[output_c:output_c*2]
            std = jnp.exp(log_std)
            return act, std
        
    def deterministic_forward(self, input: jnp.ndarray) -> jnp.ndarray:
        act, _ = self.forward(input=input)
        return act
    
    def stochastic_forward(self, input: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        mu, std = self.forward(input=input)
        act = mu + jax.random.normal(key=key,shape=mu.shape) * std
        return act
    
    def __repr__(self):
        return "ActorNN(\n" + super().__repr__() + ")"
    
    def init_radom_params(self, key, default_gain = 1):
        return super().init_radom_params(key, default_gain)
    
    @classmethod
    def build_actor(cls,
                    obs_dim: int,
                    act_dim: int,
                    hidden_layers_count: int = 2,
                    hidden_dims: int = 256,
                    activation_func: Optional[nn.ActivationFunction]=None,
                    standalone_std: bool = True)->ActorNN:
        if activation_func is None:
            activation_func = nn.ReLU()
        
        modules = []
        
        if hidden_layers_count == 0:
            modules.append(nn.Linear.new(obs_dim,act_dim))
            modules.append(deepcopy(activation_func))
            return cls(modules, standalone_std)
        
        modules.append(nn.Linear.new(obs_dim,hidden_dims))
        modules.append(deepcopy(activation_func))
        
        for _ in range(hidden_layers_count - 1):
            modules.append(nn.Linear.new(hidden_dims,hidden_dims))
            modules.append(deepcopy(activation_func))
        
        modules.append(nn.Linear.new(hidden_dims,act_dim))
        
        return cls.new(*modules,standalone_std=standalone_std)

    def _tree_flatten(self):
        child = (self._modules)
        aux_data = {"standalone_std":self._standalone_std}
        return (child, aux_data)
register_pytree_node(ActorNN, ActorNN._tree_flatten, ActorNN._tree_unflatten)
    




class CriticNN(nn.Sequential):

    def __last_output_compatibility_check(self):
        """Verifies that the network configuration results in exactly one output feature."""
        if self.get_output_count(self._modules) > 1:
            raise nn.InitializationException("A Critic nn can only have one output")
        elif self.get_output_count(self._modules) == 1:
            return
        else:
            raise nn.InitializationException("A Critic nn must have at least one output")
        
    @classmethod
    def new(cls, *modules:nn.JaxModule):
        obj = cls(*modules)
        obj.__last_output_compatibility_check()
        return obj
    
    @classmethod
    def build_critic(cls:CriticNN,
                     obs_dim: int,
                     hidden_layers_count: int = 2,
                     hidden_dims: int = 256,
                     activation_func: Optional[nn.ActivationFunction]=None)->CriticNN:
        
        """Build and return a critic instance."""
        if activation_func is None:
            activation_func = nn.ReLU()
        modules = []
        
        if hidden_layers_count == 0:
            modules.append(nn.Linear.new(obs_dim,1))
            modules.append(deepcopy(activation_func))
            return cls(modules)
        
        modules.append(nn.Linear.new(obs_dim,hidden_dims))
        modules.append(deepcopy(activation_func))
        
        for _ in range(hidden_layers_count - 1):
            modules.append(nn.Linear.new(hidden_dims,hidden_dims))
            modules.append(deepcopy(activation_func))
        
        modules.append(nn.Linear.new(hidden_dims,1))
        
        return cls.new(modules)
    
register_pytree_node(CriticNN, CriticNN._tree_flatten, CriticNN._tree_unflatten)


class QNN(CriticNN):
    def list_forward(self, obs_act:list[jnp.ndarray,jnp.ndarray])->jnp.ndarray:
        """Performs the forward pass for the Q-function approximator with a list or a tuple as input.
        by convention, the order should be [observation, action].
        """
        input = jnp.concatenate(obs_act)
        return self.forward(input=input)
    
    @classmethod
    def build_Q(cls:CriticNN,
                obs_dim: int,
                act_dim: int,
                hidden_layers_count: int = 2,
                hidden_dims: int = 256,
                activation_func: Optional[nn.ActivationFunction]=None)->CriticNN:
        
        """Build and return a critic instance."""
        if activation_func is None:
            activation_func = nn.ReLU()
        modules = []
        
        if hidden_layers_count == 0:
            modules.append(nn.Linear.new(obs_dim+act_dim,1))
            modules.append(deepcopy(activation_func))
            return cls(modules)
        
        modules.append(nn.Linear.new(obs_dim+act_dim,hidden_dims))
        modules.append(deepcopy(activation_func))
        
        for _ in range(hidden_layers_count - 1):
            modules.append(nn.Linear.new(hidden_dims,hidden_dims))
            modules.append(deepcopy(activation_func))
        
        modules.append(nn.Linear.new(hidden_dims,1))
        
        return cls.new(modules)
register_pytree_node(QNN, QNN._tree_flatten, QNN._tree_unflatten)


def make_actor_critic_tuple(obs_dim: int,
                            act_dim: int,
                            hidden_layers_count: int = 2,
                            hidden_dims: int = 256,
                            activation_func: Optional[nn.ActivationFunction]=None,
                            standalone_std: bool = True,
                            key: Optional[jnp.ndarray] = None)->Tuple[ActorNN, CriticNN]:
    
    """Construct and return actor critic tuple.
    
    Returns:
        Tuple[ActorNN, CriticNN]: Newly constructed objects matching the requested
                                  configuration. With random params if `key` is given
    
    """
    actor = ActorNN.build_actor(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func, standalone_std)
    critic = CriticNN.build_critic(obs_dim, hidden_layers_count, hidden_dims, activation_func)
    if key is not None:
        key, key_ = jax.random.split(key)
        actor.init_radom_params(key)
        actor.init_radom_params(key_)
        
    return actor, critic

def make_actor_and_q_tuple(obs_dim: int,
                           act_dim: int,
                           Q_amount: int,
                           hidden_layers_count: int = 2,
                           hidden_dims: int = 256,
                           activation_func: Optional[nn.ActivationFunction]=None,
                           standalone_std: bool = True,
                           key: Optional[jnp.ndarray] = None)->tuple[ActorNN, tuple[QNN]]:
    
    """Construct and return q and actor tuple.
    
    
    Returns:
        Tuple[ActorNN, list[]]: Newly constructed objects matching the requested
                                configuration in the form of (actor, (Q1,Q2,...))
    """
    actor = ActorNN.build_actor(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func, standalone_std)
    if key is not None:
        key, key_ = jax.random.split(key)
        actor.init_radom_params(key=key_)
    
    Qs:list[QNN] = []
    for i in range(Q_amount):
        Qs.append(QNN.build_Q(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func))
        if key is not None:
            key, key_ = jax.random.split(key)
            Qs[i].init_radom_params(key=key_)
    
    return actor, tuple(Qs)


if __name__ == "__main__":
    modules = [
        nn.Linear.new(5,32),
        nn.ELU(),
        nn.Linear.new(32,32),
        nn.ELU(),
        nn.Linear.new(32,2),
        nn.ELU(),
        ]

    pi = ActorNN.new(*modules,standalone_std=True)
    array = jnp.array([1,2,3,4,5])
    key = jax.random.key(1)
    pi.init_radom_params(key=key)

    print(jax.jit(pi.forward)(array))
    print(pi)