"""Utility helpers shared by the reinforcement-learning package.

The module groups probability helpers, numerical normalization routines, console
warnings, and observation normalizers reused across algorithms.
"""

from abc import ABC, abstractmethod
from typing import override
import numpy as np
from numba import njit

LOG_2_PI = np.log(2*np.pi)
RED_BOLD = "\033[1m\033[31m"
RESET = "\033[0m"

def print_red_warning_simple(message: str):
    """Prints a simple red and bold warning message using direct ANSI codes.
    
    Args:
        message (str): The warning text to display.
    
    Returns:
        None: This method performs its work in place and returns no value.
    
    Example:
        >>> result = print_red_warning_simple(message="message")
    """
    # Print the colored part, followed by the reset code
    print(f"{RED_BOLD}WARNING: {message}{RESET}")

@njit(fastmath=True,cache=True)
def compute_tanh_log_prob(means: np.ndarray,
                          stds: np.ndarray, 
                          actions: np.ndarray,
                          raw_actions: np.ndarray)->np.ndarray:
    """Computes the log probability related to the hyperbolic tangent function.
    
    This function calculates a specific form of log probability, likely used in sampling or
    likelihood estimation within a model context. It first computes a raw log probability
    using an external function (assumed to exist, `compute_log_prob`) and then subtracts a
    correction term derived from the provided `actions` and `raw_actions` arrays.
    
    Args:
        means (np.ndarray): Array containing the mean values (mu) for the distribution.
        stds (np.ndarray): Array containing the standard deviation values (sigma). for the.
                           distribution.
        actions (np.ndarray): Array representing the action values, used in the. correction.
                              term calculation.
        raw_actions (np.ndarray): Array containing the raw action values, used by.
                                  `compute_log_prob`.
    
    Returns:
        np.ndarray: The computed log probability array after applying the correction term.
    
    Example:
        >>> result = compute_tanh_log_prob(means=..., stds=..., actions=..., ...)
    """
    log_prob_raw = compute_log_prob(means=means,stds=stds, actions=raw_actions)
    correction_term = np.sum(np.log(1-(actions*actions) + 1e-6),axis=1)
    return log_prob_raw - correction_term

@njit(fastmath=True,cache=True)
def compute_log_prob(means: np.ndarray, stds: np.ndarray, actions: np.ndarray):
    """Computes the log probability of actions given Gaussian distributions.
    
    This method is JIT-compiled with Numba for high performance.
    
    Args:
        means (np.ndarray): The means of the Gaussian distributions.
        stds (np.ndarray): The standard deviations of the Gaussian distributions.
        actions (np.ndarray): The actual actions taken.
    
    Returns:
        Any: The computed log probabilities for the actions.
    
    Example:
        >>> result = compute_log_prob(means=..., stds=..., actions=...)
    """
    diff_sqrd = (actions-means)**2
    inner_sum = diff_sqrd/stds**2 + 2*np.log(stds)
    summed = np.sum(inner_sum,axis=1)
    log_probs = (-0.5) * (summed + actions.shape[1] * LOG_2_PI)
    return log_probs

def normalize_values(scores:np.ndarray)->np.ndarray:
    """Normalizes an array of scores to have a mean of 0 and a standard deviation of 1.
    
    The computation is written to be reused by multiple reinforcement-learning algorithms
    throughout the package.
    
    Args:
        scores (np.ndarray): A NumPy array containing the raw scores.
    
    Returns:
        np.ndarray: The normalized scores array.
    
    Example:
        >>> result = normalize_values(scores=...)
    """
    mean = np.mean(scores)
    std = np.std(scores)
    scores_normalized = (scores - mean)/(std+1e-9) # the 1e-9 to make sure theres no /0
    return scores_normalized


class Normalizer(ABC):
    """Class that encapsulates the normalizer behavior.
    """


    @abstractmethod
    def update_mean_std(self,data: np.ndarray):
        """Update mean std.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = Normalizer(...)
            >>> result = obj.update_mean_std(data=...)
        """
        ...
    @abstractmethod
    def normalize(self,data: np.ndarray) -> np.ndarray :
        """Normalize the provided data.
        
        The computation is written to be reused by multiple reinforcement-learning algorithms
        throughout the package.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            np.ndarray: Normalized version of the provided data.
        
        Example:
            >>> obj = Normalizer(...)
            >>> result = obj.normalize(data=...)
        """
        ...
    def normalize_and_update(self,data: np.ndarray) -> np.ndarray :
        """Normalize and update.
        
        The computation is written to be reused by multiple reinforcement-learning algorithms
        throughout the package.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            np.ndarray: Normalized data after the running statistics are updated.
        
        Example:
            >>> obj = Normalizer(...)
            >>> result = obj.normalize_and_update(data=...)
        """
        self.update_mean_std(data=data)
        return self.normalize(data=data)

class NoNormalizer(Normalizer):
    """Class that encapsulates the no normalizer behavior.
    
    Example:
        >>> obj = NoNormalizer()
    """


    @override
    def update_mean_std(self, data: np.ndarray):
        """Update mean std.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = NoNormalizer(...)
            >>> result = obj.update_mean_std(data=...)
        """
        ...
    @override
    def normalize(self, data: np.ndarray) -> np.ndarray :
        """Normalize the provided data.
        
        The computation is written to be reused by multiple reinforcement-learning algorithms
        throughout the package.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            np.ndarray: Normalized version of the provided data.
        
        Example:
            >>> obj = NoNormalizer(...)
            >>> result = obj.normalize(data=...)
        """
        return data
    
    @override
    def normalize_and_update(self, data: np.ndarray) -> np.ndarray :
        """Normalize and update.
        
        The computation is written to be reused by multiple reinforcement-learning algorithms
        throughout the package.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            np.ndarray: Normalized data after the running statistics are updated.
        
        Example:
            >>> obj = NoNormalizer(...)
            >>> result = obj.normalize_and_update(data=...)
        """
        return data


class RunningMeanStd(Normalizer):
    """based on this wikipedia arcticle: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    
    Args:
        data_length (int): Expected feature dimension of the running statistics.
    
    Example:
        >>> obj = RunningMeanStd(data_length=4)
    """


    def __init__(self, data_length: int, normalized_obs:np.ndarray=None):
        #! all the type checking made by Gemini
        # Type and Value checking for data_length
        """Initialize a new RunningMeanStd instance.
        
        Args:
            data_length (int): Expected feature dimension of the running statistics.
            normalized_obs (ndarray): Mask to determine which observation(s) to normalize
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            ValueError: Raised if one or more argument values are invalid or unsupported.
        """
        if not isinstance(data_length, int):
            raise TypeError(f"data_length must be an integer, got {type(data_length).__name__}.")
        if data_length <= 0:
            raise ValueError(f"data_length must be strictly positive, got {data_length}.")
        if not isinstance(normalized_obs, (type(None), np.ndarray)):
            raise TypeError(f"normalized_obs must be of type np.ndarray or None, got {type(normalized_obs).__name__}.")
        if isinstance(normalized_obs, np.ndarray) and normalized_obs.shape != (data_length,):
            raise ValueError(f"Expected normalized_obs shape to be {(data_length,)}, but got {normalized_obs.shape}.")
        if normalized_obs is None:
            normalized_obs = np.ones(data_length, dtype=np.bool_)
        self.normalized_obs = normalized_obs
        self.__data_length = data_length  # saved for dimension verification
        self.__M2 = np.zeros(data_length, dtype=np.float32)
        self.__mean = np.zeros(data_length, dtype=np.float32)
        self.__count = 0

    @property
    def M2(self):
        """Return the current m2.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__M2

    @M2.setter
    def M2(self, value: np.ndarray):
        # Type checking for M2
        """Set the m2 used by this RunningMeanStd.
        
        Args:
            value (np.ndarray): Value to assign or propagate through the current operation.
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            ValueError: Raised if one or more argument values are invalid or unsupported.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"M2 value must be a numpy.ndarray, got {type(value).__name__}.")
        # Shape checking for M2
        if value.shape != (self.__data_length,):
            raise ValueError(f"Expected M2 shape to be {(self.__data_length,)}, but got {value.shape}.")
        self.__M2 = value

    @property
    def mean(self):
        """Return the current mean.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__mean

    @mean.setter
    def mean(self, value: np.ndarray):
        # Type checking for mean
        """Set the mean used by this RunningMeanStd.
        
        Args:
            value (np.ndarray): Value to assign or propagate through the current operation.
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            ValueError: Raised if one or more argument values are invalid or unsupported.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"mean value must be a numpy.ndarray, got {type(value).__name__}.")
        # Shape checking for mean
        if value.shape != (self.__data_length,):
            raise ValueError(f"Expected mean shape to be {(self.__data_length,)}, but got {value.shape}.")
        self.__mean = value

    @property
    def count(self):
        """Return the current count.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.__count

    @count.setter
    def count(self, value: int):
        # Type and Value checking for count
        """Set the count used by this RunningMeanStd.
        
        Args:
            value (int): Value to assign or propagate through the current operation.
        
        Raises:
            TypeError: Raised if one or more arguments do not match the expected types.
            ValueError: Raised if one or more argument values are invalid or unsupported.
        """
        if not isinstance(value, int):
            raise TypeError(f"count must be an integer, got {type(value).__name__}.")
        if value < 0:
            raise ValueError(f"count cannot be negative, got {value}.")
        self.__count = value

    @property
    def var(self):
        """Return the current var.
        
        Returns:
            Any: Computed result returned by the function.
        """
        if self.__count == 0:
            return np.ones_like(self.__M2)
        return self.__M2 / self.__count
    
    @property
    def std(self):
        """Return the current std.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self.var**0.5
        
    @override
    def update_mean_std(self, data: np.ndarray):
        # if it's the first batch of data, you cant just do the maths with count as zero 
        # so we just calculate the M2 and the mean of the data
        """Update mean std.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = RunningMeanStd(...)
            >>> result = obj.update_mean_std(data=...)
        """
        if self.__count == 0:
            self.__M2 = np.var(data, axis=0) * data.shape[0]
            self.__mean = np.mean(data, axis=0)
            self.__count = data.shape[0]
            return
        
        new_count = self.__count + data.shape[0]
        delta = np.mean(data, axis=0) - self.__mean
        self.__mean += delta * (data.shape[0] / new_count)
        data_M2 = np.var(data, axis=0) * data.shape[0]
        self.__M2 += data_M2 + (delta**2 * self.__count * data.shape[0]) / new_count
        self.__count = new_count
        
    @override
    def normalize(self, data: np.ndarray):
        """Normalize the provided data.
        
        The computation is written to be reused by multiple reinforcement-learning algorithms
        throughout the package.
        
        Args:
            data (np.ndarray): Input data used to update or normalize the running statistics.
        
        Returns:
            Any: Normalized version of the provided data.
        
        Example:
            >>> obj = RunningMeanStd(...)
            >>> result = obj.normalize(data=...)
        """
        data = np.where(self.normalized_obs, (data - self.__mean) / (self.std + 1e-8), data)
        return data
    
    def state(self):
        """Returns the current state of the normalizer for checkpointing.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Returns:
            Any: Serializable snapshot of the current internal state.
        
        Example:
            >>> obj = RunningMeanStd(...)
            >>> result = obj.state()
        """
        return {
            "mean": self.mean.copy() if hasattr(self.mean, "copy") else self.mean,
            "M2": self.M2.copy() if hasattr(self.M2, "copy") else self.var,
            "count": self.count
        }

    def load_state(self, state):
        """Loads a saved state back into the normalizer.
        
        The serialized payload is intended for checkpointing so training can resume without
        rebuilding the object's internal runtime state from scratch.
        
        Args:
            state (Any): Serialized state previously produced by `state()`.
        
        Returns:
            None: This method updates the object in place and returns no value.
        
        Example:
            >>> obj = RunningMeanStd(...)
            >>> result = obj.load_state(state=...)
        """
        self.mean = state["mean"].copy() if hasattr(state["mean"], "copy") else state["mean"]
        self.M2 = state["M2"].copy() if hasattr(state["M2"], "copy") else state["M2"]
        self.count = state["count"]
        