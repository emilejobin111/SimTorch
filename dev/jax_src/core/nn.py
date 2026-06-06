from __future__ import annotations
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import override, Iterator
import time


class JaxModule(ABC):
    @abstractmethod
    def forward(self, input: jnp.ndarray)->jnp.ndarray:
        pass
    @abstractmethod
    def _tree_flatten(self):
        pass
    @abstractmethod
    def params(self) -> Iterator[jnp.ndarray]:
        pass
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class ActivationFunction(JaxModule):
    # @abstractmethod
    def init_weight_function(self, module:JaxModule):
        pass
    def params(self): # to overide if the activation layer has params
        yield from []
    def _tree_flatten(self):
        return ((), {})


class Sigmoid(ActivationFunction):
    def __repr__(self):
        return "Sigmoid()"
    
    @override
    @jax.jit
    def forward(self, input: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(input)
jax.tree_util.register_pytree_node(Sigmoid,Sigmoid._tree_flatten,Sigmoid._tree_unflatten)


class Tanh(ActivationFunction):
    def __repr__(self):
        return "Tanh()"
    
    @override
    @jax.jit
    def forward(self, input: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.tanh(input)
jax.tree_util.register_pytree_node(Tanh,Tanh._tree_flatten,Tanh._tree_unflatten)


class ReLU(ActivationFunction):
    def __repr__(self):
        return "ReLU()"
    
    @override
    @jax.jit
    def forward(self, input: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(input)
jax.tree_util.register_pytree_node(ReLU,ReLU._tree_flatten,ReLU._tree_unflatten)


class ELU(ActivationFunction):
    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
    @jax.jit
    def forward(self, input):
        return jax.nn.elu(input, self._alpha)
    @property
    def alpha(self):
        return self._alpha
    def __repr__(self):
        return f"ELU(alpha={self._alpha})"
    def _tree_flatten(self):
        return ((), {"alpha":self._alpha})
jax.tree_util.register_pytree_node(ELU,ELU._tree_flatten,ELU._tree_unflatten)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = -0.01):
        """Initialize a new ELU instance.
        
        Args:
            alpha (float): Scaling or learning-rate coefficient used by the routine.
        """
        self._alpha = alpha
    @jax.jit
    def forward(self, input):
        return jax.nn.leaky_relu(input, -self._alpha)
    @property
    def alpha(self):
        return self._alpha
    def __repr__(self):
        return f"LeakyReLU(alpha={self._alpha})"
    def _tree_flatten(self):
        return ((), {"alpha":self._alpha})
jax.tree_util.register_pytree_node(LeakyReLU,LeakyReLU._tree_flatten,LeakyReLU._tree_unflatten)


class Linear(JaxModule):
    def __init__(self, W:jnp.ndarray, b:jnp.ndarray, use_bias:bool):
        self._W = W
        self._b = b
        self._use_bias = use_bias
    @classmethod
    def new(cls, input_count: int, output_count: int, use_bias:bool=True):
        W = jnp.zeros((input_count, output_count))
        b = jnp.zeros((output_count))
        return cls(W,b,use_bias)
    def __repr__(self):
        return f"Linear(input_count={self._W.shape[0]},output_count={self._W.shape[1]})"
    
    def params(self):
        yield self._W
        yield self._b
    def forward(self, input):
        return input @ self._W + (self._b if self._use_bias else 0)
    
    def _tree_flatten(self):
        child = (self._W, self._b)
        aux_data = {"use_bias":self._use_bias}
        return (child, aux_data)
jax.tree_util.register_pytree_node(Linear,Linear._tree_flatten,Linear._tree_unflatten)
    
class Sequential(JaxModule):
    def __init__(self, *modules:JaxModule):
        self._modules = modules
        
    def __repr__(self):
        string = "Sequential([\n"
        for module in self._modules:
            string += f"    {repr(module)},\n"
        string += "])"
        return string
    
    def params(self):
        for module in self._modules:
            yield from module.params()
    @jax.jit
    def forward(self, input):
        for module in self._modules:
            input = module.forward(input)
        return input
    def _tree_flatten(self):
        return (self._modules,{})
jax.tree_util.register_pytree_node(Sequential,Sequential._tree_flatten,Sequential._tree_unflatten)

@jax.value_and_grad
def mse(module:JaxModule, x:jnp.ndarray, y:jnp.ndarray):
    loss = jnp.mean((module.forward(x) - y)**2)
    return loss




if __name__ == "__main__":
    
    x = jnp.array([-1,-0.01,0.01,1])
    y = jnp.array([1,2,3,8])
    
    mod = Sigmoid()
    print(mod.forward(x))
    
    mods = (Tanh(),Linear.new(4,4),ReLU(),ELU())
    network = Sequential(*mods)
    print(network)
    loss, grad_network = mse(network,x,y)
    grad_network:JaxModule
    print(loss)
    [print(x) for x in grad_network.params()]
    [print(x) for x in network.params()]
    
    import torch.nn as nn
    nn.ReLU().parameters()
    
    
    
    