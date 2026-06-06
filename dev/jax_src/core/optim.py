from __future__ import annotations
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import override, Iterator
import time
from .nn import JaxModule


class AdamW:
    """AdamW (Adam with Weight Decay) optimizer."""
    
    
    def __init__(self, m:jnp.ndarray, v:jnp.ndarray, t:jnp.ndarray,
                 maximise: bool, alpha: float, weight_decay: float,
                 grad_clip: float, beta1: float, beta2: float, epsilon: float):
        self.__maximise = maximise
        self.__alpha = alpha
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon
        self.__grad_clip = grad_clip
        self.__weight_decay = weight_decay
        
        self.__m = m
        self.__v = v
        self.__t = t
    @classmethod
    def new(cls, net:JaxModule,
            maximise: bool = True, alpha: float = 0.01, weight_decay: float = 0.0,
            grad_clip: float = float("inf"), beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        m = jax.tree.map(jnp.zeros_like, net)
        v = jax.tree.map(jnp.zeros_like, net)
        
        return cls(m=m,v=v,t=jnp.array(1),
                   maximise=maximise,alpha=alpha,weight_decay=weight_decay,
                   grad_clip=grad_clip,beta1=beta1,beta2=beta2,epsilon=epsilon)
        
        
    def step(self, net:JaxModule, grad_net:JaxModule)->tuple[JaxModule,AdamW]:
        updates_tree = jax.tree.map(self.adamw_func,net,grad_net,self.__m,self.__v)
        
        new_net = jax.tree.map(lambda x: x[0], updates_tree)
        m = jax.tree.map(lambda x: x[1], updates_tree)
        v = jax.tree.map(lambda x: x[2], updates_tree)
        t = self.__t + 1
        
        child = (m,v,t)
        aux_data = self._tree_flatten()[1]
        new_optim = AdamW._tree_unflatten(aux_data=aux_data,children=child)
        
        return new_net, new_optim

    def adamw_func(self,param:jnp.ndarray,gradient:jnp.ndarray, m:jnp.ndarray,v:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gradient_norm = jnp.linalg.norm(gradient)
        
        gradient = jnp.where(gradient_norm > self.__grad_clip,
                  self.__grad_clip * (gradient / (gradient_norm + 1e-6)),
                  gradient)
        
        new_m = self.__beta1 * m + (1 - self.__beta1) * gradient
        new_v = self.__beta2 * v + (1 - self.__beta2) * gradient**2
        m_hat = new_m / (1 - self.__beta1**self.__t)
        v_hat = new_v / (1 - self.__beta2**self.__t)
        
        new_param = param
        if self.__weight_decay > 0.0 and param.ndim > 1:
            new_param = new_param - (self.__alpha * self.__weight_decay * param)
        
        update = self.__alpha * (m_hat / (jnp.sqrt(v_hat) + self.__epsilon))
        if self.__maximise:
            new_param = new_param + update
        else:
            new_param = new_param - update
        return new_param, new_m, new_v

    
    
    def _tree_flatten(self):
        child = (self.__m, self.__v,self.__t)
        aux_data = {"maximise":self.__maximise, "alpha":self.__alpha, "weight_decay":self.__weight_decay,
                    "grad_clip":self.__grad_clip,"beta1":self.__beta1,"beta2":self.__beta2,"epsilon":self.__epsilon}
        return (child,aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
jax.tree_util.register_pytree_node(AdamW,AdamW._tree_flatten,AdamW._tree_unflatten)
    