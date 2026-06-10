from __future__ import annotations
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import override
from jax.tree_util import register_pytree_node

class Normalizer(ABC):
    """Class that encapsulates the normalizer behavior.
    """
    @classmethod
    @abstractmethod
    def new(cls, data_length: int, normalized_obs:jnp.ndarray=None)->Normalizer:
        ...
    @abstractmethod
    def update_mean_std(self,data: jnp.ndarray)->Normalizer:
        ...
    @abstractmethod
    def normalize(self,data: jnp.ndarray) -> jnp.ndarray :
        ...
    @abstractmethod
    def normalize_and_update(self,data: jnp.ndarray) -> tuple[jnp.ndarray,Normalizer] :
        new_norm = self.update_mean_std(data=data)
        normalized_data = new_norm.normalize(data=data)
        return normalized_data, new_norm
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class NoNormalizer(Normalizer):
    def __init__(self):
        pass
    
    @classmethod
    @override
    def new(cls, data_length: int, normalized_obs:jnp.ndarray=None)->Normalizer:
        return cls()
    @override
    def update_mean_std(self, data:jnp.ndarray)->NoNormalizer:
        return self
    
    @override
    def normalize(self, data:jnp.ndarray)->jnp.ndarray:
        return data
    
    def _tree_flatten(self):
        return ((), {})
register_pytree_node(NoNormalizer, NoNormalizer._tree_flatten, NoNormalizer._tree_unflatten)


class RunningMeanStd(Normalizer):
    """based on this wikipedia arcticle: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm"""
    def __init__(self, M2:jnp.ndarray, mean:jnp.ndarray, count:jnp.ndarray, normalized_obs:tuple[bool],):
        """shouldn't be called, if you want to make a new instance of `RunningMeanStd`, call `RunningMeanStd.new` """
        self.M2 = M2
        self.mean = mean
        self.count = count
        self.normalized_obs = normalized_obs
    
    @classmethod
    def new(cls, data_length: int, normalized_obs:jnp.ndarray=None):
        if not isinstance(data_length, int):
            raise TypeError(f"data_length must be an integer, got {type(data_length).__name__}.")
        if data_length <= 0:
            raise ValueError(f"data_length must be strictly positive, got {data_length}.")
        if not isinstance(normalized_obs, (type(None), jnp.ndarray)):
            raise TypeError(f"normalized_obs must be of type np.ndarray or None, got {type(normalized_obs).__name__}.")
        if isinstance(normalized_obs, jnp.ndarray) and normalized_obs.shape != (data_length,):
            raise ValueError(f"Expected normalized_obs shape to be {(data_length,)}, but got {normalized_obs.shape}.")
        if normalized_obs is None:
            normalized_obs = tuple([True for _ in range(data_length)])
        else:
            normalized_obs = tuple([bool(x) for x in normalized_obs])
        M2 = jnp.zeros(data_length, dtype=jnp.float32)
        mean = jnp.zeros(data_length, dtype=jnp.float32)
        count = jnp.array(0)
        return cls(M2=M2,mean=mean,count=count,normalized_obs=normalized_obs)
    
    @override
    def update_mean_std(self, data:jnp.ndarray)->RunningMeanStd:
        new_count = self.count + data.shape[0]
        delta = jnp.mean(data, axis=0) - self.mean
        new_mean = self.mean + delta * (data.shape[0] / new_count)
        data_M2 = jnp.var(data, axis=0) * data.shape[0]
        new_M2 = self.M2 + data_M2 + (delta**2 * self.count * data.shape[0]) / new_count
        return RunningMeanStd(M2=new_M2,mean=new_mean,count=new_count,normalized_obs=self.normalized_obs)
    
    @override
    def normalize(self, data: jnp.ndarray):
        safe_count = jnp.maximum(self.count,1)
        norm_obs = jnp.array(self.normalized_obs)
        data = jnp.where(norm_obs, (data - self.mean) / ((self.M2/safe_count)**0.5 + 1e-8), data)
        return data
    
    def _tree_flatten(self):
        child = (self.M2, self.mean, self.count,)
        aux_data = {"normalized_obs":self.normalized_obs}
        return (child, aux_data)
register_pytree_node(RunningMeanStd,RunningMeanStd._tree_flatten,RunningMeanStd._tree_unflatten)
