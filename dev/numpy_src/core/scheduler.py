"""Schedulers for dynamically updating training values over time or score.

These classes bind to either NumPy buffer entries or object attributes and adjust them
according to linear, exponential, milestone-based, or score-driven strategies.
"""

from __future__ import annotations
from typing import override, Tuple, Optional, Any
from dataclasses import dataclass
import os
from abc import ABC, abstractmethod
from copy import deepcopy
import time


import numpy as np




class Scheduler(ABC):
    """Abstract base class for all schedulers.
    
    A scheduler dynamically updates a specific value over time or based on a score. The
    target value can either be an element within a NumPy array or a property of a Python
    object.
    
    Args:
        start_value (Optional[float]): Initial value controlled by the scheduler.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                              to. an object attribute.
    """
    def __init__(self,
                 start_value: Optional[float] = None, 
                 param_and_index:Optional[Tuple[np.ndarray,int]]=None, 
                 object_and_property_name: Optional[Tuple[Any,str]]=None):
        """Initializes the Scheduler and binds it to a target variable.
        
        Exactly one of `param_and_index` or `object_and_property_name` must be provided.
        
        Args:
            start_value (Optional[float]): Initial value controlled by the scheduler.
            param_and_index (Optional[Tuple[np.ndarray, int]]): A tuple containing a NumPy.
                                                                array. and the index of the.
                                                                element to be modified by the.
                                                                scheduler. Defaults to None.
            object_and_property_name (Optional[Tuple[Any, str]]): A tuple containing a Python.
                                                                  object. and the string name.
                                                                  of. the attribute to be.
                                                                  modified. by the scheduler.
                                                                  Defaults to. None.
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            ValueError: If both parameters are None, if both parameters are provided,. or if.
                        the. provided tuples do not match the expected types.
        """
        #~ --- type checking ---
        if start_value is not None and not isinstance(start_value, (int, float)):
            raise TypeError(f"start_value should be a float or int got {type(start_value)}")
        if param_and_index is None and object_and_property_name is None :
            raise ValueError("only one of these two can be None : param|object_and_property_name")
        if not param_and_index is None and not object_and_property_name is None:
            raise ValueError("only one of these two can be None : param|object_and_property_name")
        if not isinstance(param_and_index, tuple) and not isinstance(object_and_property_name, tuple):
            raise ValueError("param_and_index and object_and_property_name should be a tuple")
        if param_and_index is not None:
            if not isinstance(param_and_index[0], np.ndarray) and not isinstance(param_and_index[1], int):
                raise ValueError("param_and_index should be a tuple of (np.ndarray, int)")
        if object_and_property_name is not None:
            if not isinstance(object_and_property_name[0], object) and not isinstance(object_and_property_name[1], str) :
                raise ValueError("object_and_property_name should be a tuple of (object, str)")
        #~ ---------------------
        
        if isinstance(param_and_index, tuple):
            self.__param = param_and_index[0]
            self.__index = param_and_index[1]
            self.__param_mode = True
        else:
            self._object = object_and_property_name[0]
            self._property_name = object_and_property_name[1]
            self.__param_mode = False
        if start_value is not None:
            self.value = start_value
        else:
            self.value = self.value  # initialize value from current state
        
    @abstractmethod
    def step(self,score: float):
        """Advances the scheduler by one step and updates the bound value.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            score (float): Current score or metric used by the routine.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = Scheduler(...)
            >>> result = obj.step(score=0.1)
        """
        pass

    def state(self) -> dict:
        """Returns the scheduler's runtime state as a serializable dict.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = Scheduler(...)
            >>> result = obj.state()
        """
        return {}

    def load_state(self, state: dict) -> None:
        """Restores the scheduler's runtime state from a dict.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = Scheduler(...)
            >>> result = obj.load_state(state=...)
        """
        pass

    @property
    def value(self):
        """Gets the current value of the bound parameter or property.
        
        Returns:
            Any: The current value.
        """
        if self.__param_mode:
            return self.__param[self.__index]
        else:
            return getattr(self._object, self._property_name)

    @value.setter
    def value(self, value: float):
        """Sets a new value to the bound parameter or property.
        
        Args:
            value (float): The new value to apply.
        """
        if self.__param_mode:
            self.__param[self.__index] = value
        else:
            setattr(self._object, self._property_name, value)


class LinearScheduler(Scheduler):
    """Schedules a value to change linearly over a specified number of steps.
    
    The value will start at `start_value` and incrementally move towards `end_value` each
    time `step()` is called. Once `num_steps` is reached, the value will remain at
    `end_value`.
    
    Args:
        start_value (float): Initial value controlled by the scheduler.
        end_value (float): Final target value for the scheduler.
        num_steps (int): Number of scheduler steps used to reach the target value.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                              to. an object attribute.
    
    Example:
        >>> obj = LinearScheduler(start_value=0.1, end_value=0.1, num_steps=4, ...)
    """
    def __init__(self, 
                 start_value: float, 
                 end_value: float, 
                 num_steps: int, 
                 param_and_index:Optional[Tuple[np.ndarray,int]]=None, 
                 object_and_property_name: Optional[Tuple[Any,str]]=None):
        """Initializes the LinearScheduler.
        
        Args:
            start_value (float): The initial value.
            end_value (float): The final target value.
            num_steps (int): The number of steps over which to linearly interpolate the value.
            param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple of (array, index) to bind.
                                                                Defaults to None.
            object_and_property_name (Optional[Tuple[Any, str]]): Tuple of (object,.
                                                                  property_name) to bind.
                                                                  Defaults to None.
        """
        Scheduler.__init__(self, start_value, param_and_index, object_and_property_name)
        self.__start_value = start_value
        self.__end_value = end_value
        self.__t_max = num_steps
        self.__t = 0
        self.__delta = (end_value - start_value) / num_steps
    @property
    def object_and_property_name(self):
        """Gets the object and property name tuple if the scheduler is bound to an object attribute.
        
        Returns:
            Optional[Tuple[Any, str]]: The (object, property_name) tuple if bound to an object attribute, else None.
        """
        if not self.__param_mode:
            return (self._object, self._property_name)
        return None
    @property
    def param_and_index(self):
        """Gets the parameter and index tuple if the scheduler is bound to a NumPy array entry.
        
        Returns:
            Optional[Tuple[np.ndarray, int]]: The (array, index) tuple if bound to a NumPy array entry, else None.
        """
        if self.__param_mode:
            return (self.__param, self.__index)
        return None
    @property
    def start_value(self):
        """Gets the initial value of the scheduler.
        
        Returns:
            float: The initial value.
        """
        return self.__start_value
    @property
    def end_value(self):
        """Gets the final target value of the scheduler.
        
        Returns:
            float: The final target value.
        """
        return self.__end_value
    @property
    def num_steps(self):
        """Gets the number of steps over which the scheduler transitions from start to end value.
        
        Returns:
            int: The number of steps for the linear transition.
        """
        return self.__t_max
    @property
    def t(self):
        """Gets the current step count of the scheduler.
        
        Returns:
            int: The current step count.
        """
        return self.__t
    @t.setter
    def t(self, value: int):
        """Sets the current step count of the scheduler.
        
        Args:
            value (int): The new step count to set.
        """
        if not isinstance(value, int):
            raise TypeError("t should be an integer")
        if value < 0:
            raise ValueError("t should be non-negative")
        if value > self.__t_max:
            value = self.__t_max
        self.__t = value
    @override
    def step(self, score: float):
        """Advances the scheduler by one step, linearly updating the bound value.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            score (float): The current score. Unused in this time-based scheduler,. but.
                           required. by the base class signature.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = LinearScheduler(...)
            >>> result = obj.step(score=0.1)
        """
        if self.__t_max == 0:
            new_value = self.__end_value
            self.__t += 1
            self.value = new_value
        else:
            if self.__t < self.__t_max:
                new_value = self.__start_value + self.__delta * self.__t
                self.__t += 1
                self.value = new_value
            else:
                self.value = self.__end_value

    def state(self) -> dict:
        """Return a serializable snapshot of the current internal state.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = LinearScheduler(...)
            >>> result = obj.state()
        """
        return {"t": self._LinearScheduler__t}

    def load_state(self, state: dict) -> None:
        """Restore the internal state of this object from serialized data.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = LinearScheduler(...)
            >>> result = obj.load_state(state=...)
        """
        self._LinearScheduler__t = int(state["t"])


class ExponentialScheduler(Scheduler):
    """Schedules a value to decay exponentially over time.
    
    The value decays by a factor of `gamma` at each step, asymptotically approaching
    `min_possible`.
    
    Args:
        gamma (float): Discount or exponential-decay factor used by the algorithm.
        min_possible (float): Value supplied for `min_possible`.
        start_value (Optional[float]): Initial value controlled by the scheduler.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                              to. an object attribute.
    
    Example:
        >>> obj = ExponentialScheduler(gamma=0.1, min_possible=..., start_value=0.1, ...)
    """
    def __init__(self,
                 gamma: float = 0.97, 
                 min_possible: float = 0.0,
                 start_value: Optional[float] = None,
                 param_and_index:Optional[Tuple[np.ndarray,int]]=None, 
                 object_and_property_name: Optional[Tuple[Any,str]]=None):
        """Initializes the ExponentialScheduler.
        
        Args:
            gamma (float): The decay rate. Must be between 0 (exclusive) and 1 (inclusive).
                           Defaults to 0.97.
            min_possible (float): The absolute minimum value the scheduler can reach. Defaults.
                                  to 0.0.
            start_value (Optional[float]): The initial value. Defaults to None. If None, it.
                                           will. be initialized from the current state of the.
                                           bound. variable.
            param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple of (array, index) to bind.
                                                                Defaults to None.
            object_and_property_name (Optional[Tuple[Any, str]]): Tuple of (object,.
                                                                  property_name) to bind.
                                                                  Defaults to None.
        
        Raises:
            TypeError: If gamma or min_possible are not floats.
            ValueError: If gamma is not within the valid range (0, 1].
        """
        Scheduler.__init__(self, start_value, param_and_index, object_and_property_name)
        if not isinstance(gamma,float):
            raise TypeError("gamma needs to be a float")
        if not isinstance(min_possible,float):
            raise TypeError("min_possible needs to be a float")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma needs to be between 0 (excl) and 1 (incl)")
        if self.value < min_possible :
            self.value = min_possible
        
        self._gamma = gamma
        self._min_possible = min_possible

    @override
    def step(self, Score: float):
        """Advances the scheduler by one step, exponentially decaying the bound value.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            Score (float): The current score. Unused in this time-based scheduler,. but.
                           required. by the base class signature.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = ExponentialScheduler(...)
            >>> result = obj.step(Score=...)
        """
        self.value = (self.value - self._min_possible) * self._gamma + self._min_possible

    def state(self) -> dict:
        """Return a serializable snapshot of the current internal state.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = ExponentialScheduler(...)
            >>> result = obj.state()
        """
        return {"min_possible": self._min_possible}

    def load_state(self, state: dict) -> None:
        """Restore the internal state of this object from serialized data.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = ExponentialScheduler(...)
            >>> result = obj.load_state(state=...)
        """
        if "min_possible" in state:
            self._min_possible = float(state["min_possible"])


class StepScheduler(Scheduler):
    """Schedules a value to change at specific step milestones.
    
    Provided a 2D array of `[steps, values]`, the bound value will jump to the defined value
    once the internal step counter reaches or exceeds the corresponding step milestone.
    
    Args:
        steps_and_values (np.typing.ArrayLike): Step-to-value table used by a milestone.
                                                scheduler.
        start_value (Optional[float]): Initial value controlled by the scheduler.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[object, str]]): Tuple binding the.
                                                                 scheduler. to an object.
                                                                 attribute.
    
    Example:
        >>> obj = StepScheduler(steps_and_values=..., start_value=0.1, param_and_index=..., ...)
    """
    def __init__(self,
                 steps_and_values:np.typing.ArrayLike,
                 start_value: Optional[float] = None,
                 param_and_index:Optional[Tuple[np.ndarray,int]]=None, 
                 object_and_property_name: Optional[Tuple[object,str]]=None):
        """Initializes the StepScheduler.
        
        Args:
            steps_and_values (np.typing.ArrayLike): A 2D array-like structure where the first.
                                                    row. represents the step milestones.
                                                    (positive integers) and the second row.
                                                    represents the values.
            start_value (Optional[float]): The initial value. Defaults to None. If None, it.
                                           will. be initialized from the current state of the.
                                           bound. variable.
            param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple of (array, index) to bind.
                                                                Defaults to None.
            object_and_property_name (Optional[Tuple[object, str]]): Tuple of (object,.
                                                                     property_name) to bind.
                                                                     Defaults to None.
        
        Raises:
            ValueError: If `steps_and_values` cannot be reshaped into a 2D array, or if any.
                        provided step milestones are negative.
        """
        Scheduler.__init__(self, start_value, param_and_index, object_and_property_name)
        steps_and_values = np.array(steps_and_values)
        steps_and_values = steps_and_values.reshape(2,-1)
        #~ --- type checking ---
        if not steps_and_values.ndim == 2:
            raise ValueError("steps_and_values should be a 2D array")
        #~ ---------------------
        sorted_indexs = np.argsort(steps_and_values[0])
        
        self._values = list(steps_and_values[1][sorted_indexs])[::-1]
        self._steps = list(steps_and_values[0][sorted_indexs])[::-1]
        
        if len(self._steps) > 0 and self._steps[-1] < 0:
            raise ValueError("steps should be positive")
            
        if len(self._steps) > 0 and self._steps[-1] == 0:
            self._steps.pop()
            self.value = self._values.pop()
        
        self._t = 0
        
    def _get_target_by_steps(self):
        """Internal helper that returns target by steps.
        
        Returns:
            Any: Next scheduled value to apply, if any.
        """
        new_target = None
        while len(self._steps) > 0 and self._steps[-1] <= self._t:
            self._steps.pop()
            new_target = self._values.pop()
        return new_target
        
    def step(self, Score: float):
        """Advances the internal step counter and applies any value changes if a milestone is hit.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            Score (float): The current score. Unused in this time-based scheduler,. but.
                           required. by the base class signature.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = StepScheduler(...)
            >>> result = obj.step(Score=...)
        """
        self._t += 1
        new_value = self._get_target_by_steps()
        if new_value is not None:
            self.value = new_value

    def state(self) -> dict:
        """Return a serializable snapshot of the current internal state.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = StepScheduler(...)
            >>> result = obj.state()
        """
        return {
            "t":      self._t,
            "steps":  list(self._steps),
            "values": list(self._values),
        }

    def load_state(self, state: dict) -> None:
        """Restore the internal state of this object from serialized data.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = StepScheduler(...)
            >>> result = obj.load_state(state=...)
        """
        self._t      = int(state["t"])
        self._steps  = list(state["steps"])
        self._values = list(state["values"])


class ByScoreScheduler(Scheduler):
    """Schedules a value to change dynamically based on a score metric.
    
    Instead of changing over time, this scheduler observes the `score` provided to `step()`.
    Depending on the `score_and_values` thresholds, it updates the bound value. It can be
    configured to only trigger when beating previous best scores, or to trigger
    bidirectionally (step down) based on fluctuating scores
    
    Args:
        score_and_values (np.typing.ArrayLike): Threshold-to-value table used by a.
                                                score-based scheduler.
        step_down (bool): Whether score-based scheduling should also react to score.
                          decreases.
        start_value (Optional[float]): Initial value controlled by the scheduler.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                              to. an object attribute.
    
    Example:
        >>> obj = ByScoreScheduler(score_and_values=..., step_down=False, start_value=0.1, ...)
    """
    def __init__(self, 
                 score_and_values: np.typing.ArrayLike,
                 step_down: bool=False,
                 start_value: Optional[float] = None,
                 param_and_index:Optional[Tuple[np.ndarray,int]]=None, 
                 object_and_property_name: Optional[Tuple[Any,str]]=None):
        """Initializes the ByScoreScheduler.
        
        Args:
            score_and_values (np.typing.ArrayLike): A 2D array-like structure where the first.
                                                    row. contains score thresholds and the.
                                                    second row contains the corresponding.
                                                    target. values.
            step_down (bool): If False, the scheduler tracks the maximum score seen and ignores.
                              score drops. If True, the scheduler responds to the immediate.
                              score provided, allowing the scheduled. value to revert if the.
                              score drops. Defaults to False.
            start_value (Optional[float]): The initial value. Defaults to None. If None, it.
                                           will. be initialized from the current state of the.
                                           bound. variable.
            param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple of (array, index) to bind.
                                                                Defaults to None.
            object_and_property_name (Optional[Tuple[Any, str]]): Tuple of (object,.
                                                                  property_name) to bind.
                                                                  Defaults to None.
        
        Raises:
            ValueError: If `score_and_values` is not a 2D array.
            TypeError: If `step_down` is not a boolean.
        """
        Scheduler.__init__(self,start_value, param_and_index, object_and_property_name)
        
        score_and_values = np.array(score_and_values)
        score_and_values = score_and_values.reshape(2,-1)
        #~ --- type checking ---
        if not score_and_values.ndim == 2:
            raise ValueError("score_and_values should be a 2D array")
        if not isinstance(step_down, bool):
            raise TypeError("step_down should be a boolean")
        #~ ---------------------
        
        sorted_indexs = np.argsort(score_and_values[0])
        self._values = score_and_values[1][sorted_indexs]
        self._scores = score_and_values[0][sorted_indexs]
        self.value = self._values[0]
        self._step_down = step_down
        if not step_down:
            self._best_score = float("-inf")
            
    def _get_target_by_score(self, score: float) -> float:
        """Helper function to calculate the target value without directly assigning it.
        
        Args:
            score (float): Current score or metric used by the routine.
        
        Returns:
            float: Target value selected from the configured score thresholds.
        """
        if not self._step_down:
            self._best_score = max(score, self._best_score)
            score = self._best_score
        
        idx = np.searchsorted(self._scores, score, side='right') - 1
        idx = np.clip(idx, 0, len(self._values) - 1)
        
        return float(self._values[idx])

    @override
    def step(self, score: float):
        """Evaluates the provided score and updates the scheduled value based on the thresholds.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            score (float): The current performance score or metric.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = ByScoreScheduler(...)
            >>> result = obj.step(score=0.1)
        """
        self.value = self._get_target_by_score(score)

    def state(self) -> dict:
        """Return a serializable snapshot of the current internal state.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = ByScoreScheduler(...)
            >>> result = obj.state()
        """
        if not self._step_down:
            return {"best_score": self._best_score}
        return {}

    def load_state(self, state: dict) -> None:
        """Restore the internal state of this object from serialized data.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = ByScoreScheduler(...)
            >>> result = obj.load_state(state=...)
        """
        if not self._step_down and "best_score" in state:
            self._best_score = float(state["best_score"])


class ByScoreExponentialScheduler(ByScoreScheduler, ExponentialScheduler):
    """Schedules a value to decay exponentially, but updates the decay target (min_possible).
    
    Inherits from both ByScoreScheduler (to evaluate targets based on score) and
    ExponentialScheduler (to handle the smooth decay towards that target).
    
    Args:
        score_and_values (np.typing.ArrayLike): Threshold-to-value table used by a.
                                                score-based scheduler.
        gamma (float): Discount or exponential-decay factor used by the algorithm.
        step_down (bool): Whether score-based scheduling should also react to score.
                          decrease
        start_value (Optional[float]): Initial value controlled by the scheduler.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                              to. an object attribute.
    
    Example:
        >>> obj = ByScoreExponentialScheduler(score_and_values=..., gamma=0.1, step_down=False, ...)
    """
    def __init__(self, 
                 score_and_values: np.typing.ArrayLike,
                 gamma: float = 0.97,
                 step_down: bool = False,
                 start_value: Optional[float] = None,
                 param_and_index: Optional[Tuple[np.ndarray, int]] = None, 
                 object_and_property_name: Optional[Tuple[Any, str]] = None):
        
        # Initialize the score thresholds logic
        """Initialize a new ByScoreExponentialScheduler instance.
        
        Args:
            score_and_values (np.typing.ArrayLike): Threshold-to-value table used by a.
                                                    score-based scheduler.
            gamma (float): Discount or exponential-decay factor used by the algorithm.
            step_down (bool): Whether score-based scheduling should also react to score.
                              decreases.
            start_value (Optional[float]): Initial value controlled by the scheduler.
            param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                                a. NumPy array entry.
            object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                                  to. an object attribute.
        """
        ByScoreScheduler.__init__(self, 
                                  score_and_values=score_and_values, 
                                  step_down=step_down, 
                                  start_value=start_value,
                                  param_and_index=param_and_index, 
                                  object_and_property_name=object_and_property_name)
        
        # Initialize the exponential decay logic, using the first target as the initial minimum
        initial_min_possible = float(self._values[0])
        ExponentialScheduler.__init__(self, 
                                      gamma=gamma, 
                                      min_possible=initial_min_possible,
                                      start_value=start_value, 
                                      param_and_index=param_and_index, 
                                      object_and_property_name=object_and_property_name)

    @override
    def step(self, score: float):
        """Updates the decay target based on the score, then performs an exponential step.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            score (float): The current performance score or metric.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = ByScoreExponentialScheduler(...)
            >>> result = obj.step(score=0.1)
        """
        # 1. Update the minimum possible value dynamically based on the current score
        self._min_possible = self._get_target_by_score(score)
        
        # 2. Apply the exponential decay logic towards this new minimum
        ExponentialScheduler.step(self, score)

    def state(self) -> dict:
        """Return a serializable snapshot of the current internal state.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = ByScoreExponentialScheduler(...)
            >>> result = obj.state()
        """
        s = ByScoreScheduler.state(self)
        s["min_possible"] = self._min_possible
        return s

    def load_state(self, state: dict) -> None:
        """Restore the internal state of this object from serialized data.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = ByScoreExponentialScheduler(...)
            >>> result = obj.load_state(state=...)
        """
        ByScoreScheduler.load_state(self, state)
        if "min_possible" in state:
            self._min_possible = float(state["min_possible"])


class StepExponentialScheduler(StepScheduler, ExponentialScheduler):
    """Schedules a value to decay exponentially, but updates the decay target (min_possible).
    
    Instances operate on either a NumPy-array slot or an object attribute and update that
    target according to the configured scheduling rule.
    
    Args:
        steps_and_values (np.typing.ArrayLike): Step-to-value table used by a milestone.
                                                scheduler.
        gamma (float): Discount or exponential-decay factor used by the algorithm.
        start_value (Optional[float]): Initial value controlled by the scheduler.
        param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                            a. NumPy array entry.
        object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                              to. an object attribute.
    
    Example:
        >>> obj = StepExponentialScheduler(steps_and_values=..., gamma=0.1, start_value=0.1, ...)
    """
    
    def __init__(self, 
                 steps_and_values: np.typing.ArrayLike,
                 gamma: float = 0.97,
                 start_value: Optional[float] = None,
                 param_and_index: Optional[Tuple[np.ndarray, int]] = None, 
                 object_and_property_name: Optional[Tuple[Any, str]] = None):
        
        """Initialize a new StepExponentialScheduler instance.
        
        Args:
            steps_and_values (np.typing.ArrayLike): Step-to-value table used by a milestone.
                                                    scheduler.
            gamma (float): Discount or exponential-decay factor used by the algorithm.
            start_value (Optional[float]): Initial value controlled by the scheduler.
            param_and_index (Optional[Tuple[np.ndarray, int]]): Tuple binding the scheduler to.
                                                                a. NumPy array entry.
            object_and_property_name (Optional[Tuple[Any, str]]): Tuple binding the scheduler.
                                                                  to. an object attribute.
        """
        StepScheduler.__init__(self,
                               steps_and_values=steps_and_values,
                               start_value=start_value,
                               param_and_index=param_and_index, 
                               object_and_property_name=object_and_property_name)
        
        initial_min_possible = float(self.value)
        
        ExponentialScheduler.__init__(self, 
                                      gamma=gamma,
                                      min_possible=initial_min_possible,
                                      start_value=start_value,
                                      param_and_index=param_and_index, 
                                      object_and_property_name=object_and_property_name)
    
    @override
    def step(self, score: float):
        """Updates the decay target based on the step, then performs an exponential step.
        
        Callers typically invoke this method repeatedly as part of a training loop or scheduling
        policy.
        
        Args:
            score (float): The current performance score or metric.
        
        Returns:
            None: This method updates the current object in place and returns no value.
        
        Example:
            >>> obj = StepExponentialScheduler(...)
            >>> result = obj.step(score=0.1)
        """
        self._t += 1
        
        new_min = self._get_target_by_steps()
        if new_min is not None:
            self._min_possible = new_min
        
        ExponentialScheduler.step(self, score)

    def state(self) -> dict:
        """Return a serializable snapshot of the current internal state.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            dict: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = StepExponentialScheduler(...)
            >>> result = obj.state()
        """
        s = StepScheduler.state(self)
        s["min_possible"] = self._min_possible
        return s

    def load_state(self, state: dict) -> None:
        """Restore the internal state of this object from serialized data.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (dict): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = StepExponentialScheduler(...)
            >>> result = obj.load_state(state=...)
        """
        StepScheduler.load_state(self, state)
        if "min_possible" in state:
            self._min_possible = float(state["min_possible"])