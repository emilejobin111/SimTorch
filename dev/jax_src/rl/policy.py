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
    
    def __repr__(self):
        return "ActorMM(\n" + super().__repr__() + ")"
    
    def init_radom_params(self, key, default_gain = 1):
        return super().init_radom_params(key, default_gain)

    def _tree_flatten(self):
        child = (self._modules)
        aux_data = {"standalone_std":self._standalone_std}
        return (child, aux_data)
register_pytree_node(ActorNN, ActorNN._tree_flatten, ActorNN._tree_unflatten)







