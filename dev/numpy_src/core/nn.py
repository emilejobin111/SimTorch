"""Core neural-network architecture and modular building blocks.

This module defines the low-level abstractions and layers used to assemble trainable
models while sharing contiguous parameter and gradient buffers across an entire network.
"""



from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import override
import time

from .rng import rng



class InitializationException(Exception):
    """Custom exception raised for initialization errors in the neural network.

    

    This exception is used to flag issues such as uninitialized module indices, incompatible

    layer dimensions, or improper parameter setups before performing forward or backward

    passes.

    

    Args:

        *args (Any): Variable length argument list passed to the base Exception.

    

    Example:

        >>> obj = InitializationException(*args=...)

    """



    def __init__(self, *args):
        """Initialize a new InitializationException instance.
        
        Args:
            *args (Any): Additional arguments forwarded to the underlying implementation.
        """
        super().__init__(*args)



class NNModule(ABC):
    """Base abstract class for all neural network modules.

    

    Every layer or container in the neural network should inherit from this class. It

    enforces the implementation of standard forward and backward passes, as well as

    parameter indexing within a global array.

    

    Args:

        param_count (int): The total number of trainable parameters in the module.

    """



    def __init__(self, param_count: int):
        """Initialize a new NNModule instance.
        
        Args:
            param_count (int): Number of param used by the routine.
        """
        self._param_count = param_count
        
    @property
    def param_count(self) -> int:
        """Property to get the parameter count of the module.
        
        Returns:
            int: The number of trainable parameters in this specific module.
        """
        return self._param_count
    
    
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Performs the forward pass of the module.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The output tensor after processing.
        
        Example:
            >>> obj = NNModule(...)
            >>> result = obj.forward(input=...)
        """
        pass

    @abstractmethod
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Performs the backward pass of the module, computing gradients.
        
        The implementation uses the cached forward-pass data to propagate derivatives toward the
        previous stage and to accumulate any parameter gradients stored by the object.
        
        Args:
            dZ (np.ndarray): The gradient of the cost with respect to the output of this module.
        
        Returns:
            np.ndarray: The gradient of the cost with respect to the input of this module.
        
        Example:
            >>> obj = NNModule(...)
            >>> result = obj.backward(dZ=...)
        """
        pass
        
    @abstractmethod
    def _init_indexes(self, starting_index: int, params: np.ndarray, grads: np.ndarray) -> int:
        """Initializes the parameter indices within the global parameter array.
        
        Args:
            starting_index (int): The index where this module's parameters start.
            params (np.ndarray): The shared global parameter array.
            grads (np.ndarray): The shared global gradients array.
        
        Returns:
            int: The next available starting index after this module.
        """
        pass


class ActivationFunction(NNModule):
    """Base class for activation functions.

    

    Activation functions are treated as neural network modules but contain no learnable

    parameters.

    """



    def __init__(self):
        """Initialize a new ActivationFunction instance.
        """
        super().__init__(param_count=0)
    @override
    def _init_indexes(self, starting_index: int, params: np.ndarray, grads: np.ndarray) -> int:
        """Initializes indices for the activation function (noop since it has no parameters).
        
        Args:
            starting_index (int): The current starting index.
            params (np.ndarray): The shared global parameter array.
            grads (np.ndarray): The shared global gradients array.
        
        Returns:
            int: The unmodified starting index.
        """
        return starting_index
        
    @abstractmethod
    def init_weight_function(self, module: Linear):
        """Initializes the weights of a preceding Linear module based on the specific activation logic.
        
        Different activation functions require different weight initialization strategies (e.g.,
        He initialization for ReLU, Xavier for Tanh) to prevent vanishing or exploding
        gradients.
        
        Args:
            module (Linear): The linear layer whose weights need initialization.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = ActivationFunction(...)
            >>> result = obj.init_weight_function(module=...)
        """
        pass

class HardShrink(ActivationFunction):
    """Hard Shrinkage activation function.

    Applies the element-wise function: f(x) = x if |x| > lambda, and f(x) = 0 otherwise. This
    function is used for denoising and sparsity.

    Args:
        lambda_ (float): The threshold parameter that determines the shrinkage level.
    """
    def __init__(self, lambda_: float = 0.5):
        """Initialize a new HardShrink instance.
        
        Args:
            lambda_ (float): The threshold parameter that determines the shrinkage level.
        """
        super().__init__()
        self.__lambda = lambda_
    @property
    def lambda_(self):
        """Return the current lambda value.
        
        Returns:
            Any: The current lambda value used in the Hard Shrinkage function.
        """
        return self.__lambda
    def __repr__(self):
        """Return a compact string representation of this HardShrink.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return f"HardShrink(lambda_={self.__lambda})"
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Applies the Hard Shrinkage function element-wise and caches the input for the backward pass.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The activated output tensor with values shrunk towards zero.
        
        Example:
            >>> obj = HardShrink(...)
            >>> result = obj.forward(input=...)
        """
        self.__X_buffer = input
        return np.where(np.abs(input) > self.__lambda, input, 0)
    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Computes the gradient of the Hard Shrinkage function.
        
        The derivative is 1 for |x| > lambda, and 0 otherwise.
        
        Args:
            dZ (np.ndarray): Upstream gradients.
        
        Returns:
            np.ndarray: Gradients with respect to the input.
        
        Example:
            >>> obj = HardShrink(...)
            >>> result = obj.backward(dZ=...)
        """
        return dZ * np.where(np.abs(self.__X_buffer) > self.__lambda, 1, 0)
class Sigmoid(ActivationFunction):
    """Sigmoid activation function.

    Applies the element-wise function: f(x) = 1 / (1 + exp(-x)). Outputs are bounded
    between 0 and 1.

    Example:
        >>> obj = Sigmoid()
    """

    def __repr__(self):
        """Return a compact string representation of this Sigmoid.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return "Sigmoid()"

    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Applies the Sigmoid function element-wise and caches the result for the backward pass.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The activated output tensor with values in range (0, 1).
        
        Example:
            >>> obj = Sigmoid(...)
            >>> result = obj.forward(input=...)
        """
        self.__a_buffer = 1 / (1 + np.exp(-input))
        return self.__a_buffer

    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Computes the gradient of the Sigmoid function.
        
        The derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)).
        
        Args:
            dZ (np.ndarray): Upstream gradients.
        
        Returns:
            np.ndarray: Gradients with respect to the input.
        
        Example:
            >>> obj = Sigmoid(...)
            >>> result = obj.backward(dZ=...)
        """
        return dZ * self.__a_buffer * (1 - self.__a_buffer)

    @override
    def init_weight_function(self, module: Linear):
        """Initializes weights using Xavier/Glorot initialization suitable for Sigmoid.
        
        Draws weights from a normal distribution with mean 0 and standard deviation 
        sqrt(2 / (input_count + output_count)).
        
        Args:
            module (Linear): The linear layer whose weights need initialization.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = Sigmoid(...)
            >>> result = obj.init_weight_function(module=...)
        """
        std = (2 / (module._input_count + module._output_count)) ** 0.5
        module._W = rng.normal(0, std, size=module._W.size)
class ELU(ActivationFunction):
    """Exponential Linear Unit (ELU) activation function.

    

    Applies the element-wise function: f(x) = x if x > 0, and f(x) = alpha * (exp(x) - 1) if

    x <= 0. This function helps push the mean of the layer activations closer to zero.

    

    Args:

        alpha (float): Scaling or learning-rate coefficient used by the routine.

    

    Example:

        >>> obj = ELU(alpha=0.1)

    """



    def __init__(self, alpha: float = 1.0):
        """Initialize a new ELU instance.
        
        Args:
            alpha (float): Scaling or learning-rate coefficient used by the routine.
        """
        super().__init__()
        self.__alpha = alpha
    @property
    def alpha(self):
        """Return the current alpha value.
        
        Returns:
            Any: The current alpha value used in the ELU function.
        """
        return self.__alpha
        
    def __repr__(self):
        """Return a compact string representation of this ELU.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return f"ELU(alpha={self.__alpha})"
        
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Applies the ELU function element-wise and caches the input for the backward pass.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The activated output tensor.
        
        Example:
            >>> obj = ELU(...)
            >>> result = obj.forward(input=...)
        """
        self.__X_buffer = input
        # ELU formula: x if x > 0, else alpha * (exp(x) - 1)
        return np.where(input > 0, input, self.__alpha * (np.exp(input) - 1))
        
    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Computes the gradient of the ELU function.
        
        The derivative is 1 for x > 0, and alpha * exp(x) for x <= 0.
        
        Args:
            dZ (np.ndarray): Upstream gradients.
        
        Returns:
            np.ndarray: Gradients with respect to the input.
        
        Example:
            >>> obj = ELU(...)
            >>> result = obj.backward(dZ=...)
        """
        # Derivative: 1 if x > 0, else alpha * exp(x)
        grad_mask = np.where(self.__X_buffer > 0, 1.0, self.__alpha * np.exp(self.__X_buffer))
        return dZ * grad_mask
    
    @override
    def init_weight_function(self, module: Linear):
        """Initializes weights using a suitable initialization for ELU.
        
        Draws weights from a normal distribution with mean 0 and standard deviation sqrt(2 /
        input_count) (similar to He initialization, which works well for ELU).
        
        Args:
            module (Linear): The linear layer whose weights need initialization.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = ELU(...)
            >>> result = obj.init_weight_function(module=...)
        """
        std = (0.5 / module._input_count) ** 0.5
        module._W = rng.normal(0, std, size=module._W.size)

class Tanh(ActivationFunction):
    """Hyperbolic tangent (Tanh) activation function.
    
    Applies the element-wise function: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x). Outputs
    are bounded between -1 and 1.
    
    Example:
        >>> obj = Tanh()
    """
    def __repr__(self):
        """Return a compact string representation of this Tanh.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return "Tanh()"
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Applies the Tanh function element-wise and caches the result for the backward pass.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The activated output tensor with values in range (-1, 1).
        
        Example:
            >>> obj = Tanh(...)
            >>> result = obj.forward(input=...)
        """
        self.__a_buffer = np.tanh(input)
        return self.__a_buffer
    
    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Computes the gradient of the Tanh function.
        
        The derivative of tanh(x) is 1 - tanh^2(x).
        
        Args:
            dZ (np.ndarray): Upstream gradients.
        
        Returns:
            np.ndarray: Gradients with respect to the input.
        
        Example:
            >>> obj = Tanh(...)
            >>> result = obj.backward(dZ=...)
        """
        return dZ * (1-(self.__a_buffer)**2)
    
    @override
    def init_weight_function(self, module: Linear):
        """Initializes weights using Xavier/Glorot initialization suitable for Tanh.
        
        Draws weights from a normal distribution with mean 0 and standard deviation sqrt(2 /
        (input_count + output_count)).
        
        Args:
            module (Linear): The linear layer whose weights need initialization.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = Tanh(...)
            >>> result = obj.init_weight_function(module=...)
        """
        std = (2 / (module._input_count + module._output_count)) ** 0.5
        module._W = rng.normal(0, std, size=module._W.size)


class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function.
    
    Applies the element-wise function: f(x) = max(0, x). Outputs are bounded between 0 and
    positive infinity.
    
    Example:
        >>> obj = ReLU()
    """
    def __repr__(self):
        """Return a compact string representation of this ReLU.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return "ReLU()"
        
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Applies the ReLU function element-wise and caches the input for the backward pass.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The activated output tensor with non-negative values.
        
        Example:
            >>> obj = ReLU(...)
            >>> result = obj.forward(input=...)
        """
        self.__X_buffer = input
        return np.maximum(0, input)
        
    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Computes the gradient of the ReLU function.
        
        The derivative is 1 for x > 0, and 0 otherwise.
        
        Args:
            dZ (np.ndarray): Upstream gradients.
        
        Returns:
            np.ndarray: Gradients with respect to the input.
        
        Example:
            >>> obj = ReLU(...)
            >>> result = obj.backward(dZ=...)
        """
        return dZ * (self.__X_buffer > 0)
    
    @override
    def init_weight_function(self, module: Linear):
        """Initializes weights using He initialization suitable for ReLU.
        
        Draws weights from a normal distribution with mean 0 and standard deviation sqrt(2 /
        input_count).
        
        Args:
            module (Linear): The linear layer whose weights need initialization.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = ReLU(...)
            >>> result = obj.init_weight_function(module=...)
        """
        std = (2 / module._input_count) ** 0.5
        module._W = rng.normal(0, std, size=module._W.size)

class LeakyReLU(ActivationFunction):
    """Leaky Rectified Linear Unit activation function.
    
    Applies the element-wise function: f(x) = max(alpha * x, x). Allows a small, non-zero
    gradient when the unit is not active, preventing the "dying ReLU" problem.
    
    Args:
        alpha (float): Scaling or learning-rate coefficient used by the routine.
    
    Example:
        >>> obj = LeakyReLU(alpha=0.1)
    """
    @override
    def __init__(self, alpha: float = 0.01):
        """Initialize a new LeakyReLU instance.
        
        Args:
            alpha (float): Scaling or learning-rate coefficient used by the routine.
        """
        super().__init__()
        self.__alpha = alpha
    @property
    def alpha(self):
        """Return the current alpha value.
        
        Returns:
            Any: The current alpha value used in the LeakyReLU function.
        """
        return self.__alpha
    def __repr__(self):
        """Return a compact string representation of this LeakyReLU.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return f"LeakyReLU(alpha={self.__alpha})"
        
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Applies the Leaky ReLU function element-wise and caches the input for the backward pass.
        
        The method consumes the current inputs and cached internal state to produce the tensor
        or array expected by the next stage of the pipeline.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The activated output tensor, scaled by alpha for negative values.
        
        Example:
            >>> obj = LeakyReLU(...)
            >>> result = obj.forward(input=...)
        """
        self.__X_buffer = input
        return np.maximum(self.__alpha * input, input)
        
    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Computes the gradient of the Leaky ReLU function.
        
        The derivative is 1 for x > 0, and alpha for x <= 0.
        
        Args:
            dZ (np.ndarray): Upstream gradients.
        
        Returns:
            np.ndarray: Gradients with respect to the input.
        
        Example:
            >>> obj = LeakyReLU(...)
            >>> result = obj.backward(dZ=...)
        """
        return dZ * ((self.__X_buffer > 0) + self.__alpha * (self.__X_buffer <= 0))
    
    @override
    def init_weight_function(self, module: Linear):
        """Initializes weights using Kaiming/He initialization specifically derived for Leaky ReLU.
        
        Draws weights from a normal distribution with mean 0 and standard deviation sqrt(2 / ((1
        + alpha^2) * input_count)) to preserve variance across layers.
        
        Args:
            module (Linear): The linear layer whose weights need initialization.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Example:
            >>> obj = LeakyReLU(...)
            >>> result = obj.init_weight_function(module=...)
        """
        std = (2 / ((1 + self.__alpha**2) * module._input_count)) ** 0.5
        module._W = rng.normal(0, std, size=module._W.size)

class Linear(NNModule):
    """A fully connected (linear) neural network layer.
    
    Applies a linear transformation to the incoming data: y = x @ W + b.
    
    Args:
        input_count (int): The number of input features.
        output_count (int): The number of output features.
    
    Example:
        >>> obj = Linear(input_count=4, output_count=4)
    """
    def __init__(self, input_count: int, output_count: int):
        """Initialize a new Linear instance.
        
        Args:
            input_count (int): Number of input used by the routine.
            output_count (int): Number of output used by the routine.
        """
        self.__is_indexed = False
        self._input_count = input_count
        self._output_count = output_count
        
        param_count = input_count * output_count + output_count
        super().__init__(param_count)

    @property
    def input_count(self):
        """Return the current input count.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self._input_count

    @property
    def output_count(self):
        """Return the current output count.
        
        Returns:
            Any: Computed result returned by the function.
        """
        return self._output_count
        
        
    def __repr__(self):
        """Return a compact string representation of this Linear.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        return f"Linear({self._input_count}, {self._output_count})"
    
    @property
    def w_count(self):
        """gets the weight count of the linear layer.
        
        Returns:
            Any: the number of weight in this linear layer.
        """
        return self._input_count * self._output_count
    @property
    def b_count(self):
        """gets the bias count of the linear layer.
        
        Returns:
            Any: the number of bias in this linear layer.
        """
        return self._output_count
    
    
    @property
    def _W(self) -> np.ndarray:
        """Gets the weight matrix of the linear layer.
        
        Returns:
            np.ndarray: The weight matrix of shape (input_count, output_count).
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        if not self.__is_indexed: 
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        return self._params[self.__w_index[0]:self.__w_index[1]].reshape(self._input_count, self._output_count)
        
    @_W.setter
    def _W(self, value: np.ndarray):
        """Sets the weight matrix of the linear layer.
        
        Args:
            value (np.ndarray): The new weight values.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
            ValueError: If the provided value size does not match expected dimensions.
        """
        if not self.__is_indexed:
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        
        if value.size != (self._input_count * self._output_count):
            raise ValueError(f"Expected size {self._input_count * self._output_count}, got {value.size}")
        
        self._params[self.__w_index[0]:self.__w_index[1]] = value.flatten()
    
    @property
    def _b(self) -> np.ndarray:
        """Gets the bias vector of the linear layer.
        
        Returns:
            np.ndarray: The bias vector of shape (output_count,).
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        if not self.__is_indexed: 
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        return self._params[self.__b_index[0]:self.__b_index[1]]
        
    @_b.setter
    def _b(self, value: np.ndarray):
        """Sets the bias vector of the linear layer.
        
        Args:
            value (np.ndarray): The new bias values.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
            ValueError: If the provided value size does not match expected dimensions.
        """
        if not self.__is_indexed:
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
            
        if value.size != self._output_count:
            raise ValueError(f"Expected size {self._output_count}, got {value.size}")
            
        self._params[self.__b_index[0]:self.__b_index[1]] = value.flatten()

    @property
    def _dW(self) -> np.ndarray:
        """Gets the gradient of the weights.
        
        Returns:
            np.ndarray: The weight gradients matrix of shape (input_count, output_count).
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        if not self.__is_indexed: 
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        return self._grads[self.__w_index[0]:self.__w_index[1]].reshape(self._input_count, self._output_count)
        
    @_dW.setter
    def _dW(self, value: np.ndarray):
        """Sets the gradient of the weights.
        
        Args:
            value (np.ndarray): The new weight gradient values.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
            ValueError: If the provided value size does not match expected dimensions.
        """
        if not self.__is_indexed:
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        
        if value.size != (self._input_count * self._output_count):
            raise ValueError(f"Expected size {self._input_count * self._output_count}, got {value.size}")
        
        self._grads[self.__w_index[0]:self.__w_index[1]] = value.flatten()
    
    @property
    def _db(self) -> np.ndarray:
        """Gets the gradient of the biases.
        
        Returns:
            np.ndarray: The bias gradients vector of shape (output_count,).
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        if not self.__is_indexed: 
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        return self._grads[self.__b_index[0]:self.__b_index[1]]
        
    @_db.setter
    def _db(self, value: np.ndarray):
        """Sets the gradient of the biases.
        
        Args:
            value (np.ndarray): The new bias gradient values.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
            ValueError: If the provided value size does not match expected dimensions.
        """
        if not self.__is_indexed:
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
            
        if value.size != self._output_count:
            raise ValueError(f"Expected size {self._output_count}, got {value.size}")
            
        self._grads[self.__b_index[0]:self.__b_index[1]] = value.flatten()
    @property
    def w_slice(self)->slice:
        """Return the current w slice.
        
        Returns:
            slice: Slice describing the relevant region in the shared buffer.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        if not self.__is_indexed:
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        return slice(self.__w_index[0],self.__w_index[1])
    
    @property
    def b_slice(self)->slice:
        """Return the current b slice.
        
        Returns:
            slice: Slice describing the relevant region in the shared buffer.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
        """
        if not self.__is_indexed:
            raise InitializationException("The parameters of the Linear Module is not indexed, please call `init_index()` before anything else")
        return slice(self.__b_index[0],self.__b_index[1])
    
    @override
    def _init_indexes(self, starting_index: int, params: np.ndarray, grads: np.ndarray) -> int:
        """Initializes the starting and ending indices for weights and biases in the global arrays.
        
        Binds the local layer parameters to the shared continuous memory array.
        
        Args:
            starting_index (int): The starting position in the global array.
            params (np.ndarray): The global array holding all parameters.
            grads (np.ndarray): The global array holding all gradients.
        
        Returns:
            int: The next available index after this module's bias parameters.
        """
        
        self.__w_index = np.array([starting_index, starting_index + self._input_count * self._output_count])
        self.__b_index = np.array([self.__w_index[1], self.__w_index[1] + self._output_count])
        self._params = params
        self._grads = grads
        self.__is_indexed = True
        return self.__b_index[1]
    
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Performs the forward pass calculation (Input @ W + b).
        
        Caches the input matrix for use during the backward pass.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The output tensor after applying the linear transformation.
        
        Example:
            >>> obj = Linear(...)
            >>> result = obj.forward(input=...)
        """
        self.__X_buffer = input
        return self.__X_buffer @ self._W + self._b
    
    @override
    def backward(self, dZ):
        """Performs the backward pass, updating weight and bias gradients.
        
        Calculates gradients for weights (_dW) and biases (_db), and returns the gradient to be
        passed to the previous layer.
        
        Args:
            dZ (Any): The upstream gradient from the next layer.
        
        Returns:
            Any: The gradient of the input to be passed to the preceding layer.
        
        Example:
            >>> obj = Linear(...)
            >>> result = obj.backward(dZ=...)
        """
        self._dW += self.__X_buffer.T @ dZ
        self._db += np.sum(dZ, axis=0)
        return dZ @ self._W.T



class Sequential(NNModule):
    """A sequential container that chains multiple neural network modules together.
    
    Data passed to the forward pass of this container will travel sequentially through all
    the modules in the exact order they were passed to the constructor. By default, it acts
    as a standalone network and manages its own continuous memory allocation for parameters
    and gradients.
    
    Args:
        modules (list[NNModule]): An ordered list of neural network modules to run.
                                  sequentially.
        default_init (bool): Determines whether the container should immediately. allocate.
                             and bind its internal parameter and gradient arrays. Defaults.
                             to True. When True, the container initializes its own memory.
                             starting at index 0. When False, the memory binding is.
                             deferred. This is a critical feature for. higher-level wrapper.
                             classes (like an `Actor`, `Critic`, or a combined `Policy`.
                             class in Reinforcement Learning) that contain additional.
                             trainable parameters. outside of this sequential chain (such.
                             as. a standalone standard deviation. vector for continuous.
                             action. spaces). By deferring initialization, the parent.
                             class. can. calculate the total parameter count of all its.
                             sub-components,. allocate a single, unified global memory.
                             array, and manually. inject it by. calling `_init_indexes` on.
                             this container with. the appropriate starting offset.
    
    Example:
        >>> modules = [Relu(),Linear(12,2)]
        >>> nn = Sequential(modules=modules)
        >>> nn.init_random_params()
        >>> input = ...
        >>> result = nn.forward(input=...)
        >>> print(nn)
        Sequential([
            Relu(),
            Linear(12, 2)
        ]))
    """
    def __init__(self, modules: list[NNModule], default_init: bool = True):
        """Initialize a new Sequential instance.
        
        Args:
            modules (list[NNModule]): Ordered modules used to build a sequential network.
            default_init (bool): Whether the sequential container should initialize its buffers.
                                 immediately.
        """
        self.__module_compatibility_check(modules)
        
        param_count: int = 0
        for module in modules:
            param_count += module._param_count
        super().__init__(param_count)
        
        self._modules = modules
        
        #? --- usefull to nn that has more than just these params ---
        if default_init :
            self._params = np.zeros(shape=(self.param_count,), dtype=np.float32)
            self._grads = np.zeros(shape=(self.param_count,), dtype=np.float32)
            self._init_indexes(0, self._params, self._grads)
    
    def __module_compatibility_check(self, modules: list[NNModule]):
        """Verifies that consecutive modules have compatible input/output dimensions.
        
        Args:
            modules (list[NNModule]): The list of modules to check.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Raises:
            InitializationException: Raised if the component is used before initialization is.
                                     complete.
            NotImplementedError: If an unsupported module type is encountered during the check.
        
        Example:
            >>> obj = Sequential(...)
            >>> result = obj.__module_compatibility_check(modules=...)
        """
        last_output = None
        for module in modules:
            if isinstance(module, ActivationFunction):
                pass
            elif isinstance(module, Linear):
                if last_output is not None:
                    if last_output != module._input_count:
                        raise InitializationException("Incompatible module dimensions")
                last_output = module.output_count
            elif isinstance(module, Sequential):
                if last_output is not None:
                    if last_output != module.input_count:
                        raise InitializationException("Incompatible module dimensions")
                last_output = module.output_count
            else:
                raise NotImplementedError(f"Module compatibility check is not implemented for module: {type(module)}")

    def __repr__(self):
        """Return a compact string representation of this Sequential.
        
        Returns:
            Any: Human-readable representation of the current instance.
        """
        string = "Sequential([\n"
        for module in self._modules:
            string += f"    {repr(module)},\n"
        string += "])"
        return string
    @property
    def output_count(self):
        """Property to get the output count of the last Linear module in the sequence.
        
        Returns:
            Any: The output dimension of the final Linear layer, or None if no Linear layers
                 exist.
        """
        for module in reversed(self._modules):
            if hasattr(module, "output_count"):
                return module.output_count
        return None  # If no Linear modules are found
    @property
    def input_count(self):
        """Property to get the input count of the first Linear module in the sequence.
        
        Returns:
            Any: The input dimension of the first Linear layer, or None if no Linear layers
                 exist.
        """
        for module in self._modules:
            if hasattr(module, "input_count"):
                return module.input_count
        return None  # If no Linear modules are found
    
    @property
    def params(self) -> np.ndarray:
        """Property to get the global array containing all parameters of the network.
        
        Returns:
            np.ndarray: The flattened, continuous parameter array.
        """
        return self._params
        
    @property
    def grads(self) -> np.ndarray:
        """Property to get the global array containing all gradients of the network.
        
        Returns:
            np.ndarray: The flattened, continuous gradient array.
        """
        return self._grads
    @property
    def w_mask(self):
        """Return the current w mask.
        
        Returns:
            Any: Mask array selecting the relevant entries.
        """
        if hasattr(self,"_w_mask"):
            return self._w_mask
        else:
            self._w_mask = np.zeros(shape=self.params.shape)
            
            module: Linear
            for module in self._modules:
                if hasattr(module,"w_slice"):
                    self._w_mask[module.w_slice] = 1
            return self._w_mask
    @property
    def b_mask(self):
        """Return the current b mask.
        
        Returns:
            Any: Mask array selecting the relevant entries.
        """
        if hasattr(self,"_b_mask"):
            return self._b_mask
        else:
            self._b_mask = np.zeros(shape=self.params.shape)
            
            module: Linear
            for module in self._modules:
                if hasattr(module,"b_slice"):
                    self._b_mask[module.b_slice] = 1
            return self._b_mask
    
    @override
    def _init_indexes(self, starting_index: int, params: np.ndarray, grads: np.ndarray) -> int:
        """Sequentially sets the parameter indices for all child modules.
        
        Iterates through the container and delegates index initialization to each module,
        ensuring they map correctly to the global `params` and `grads` arrays.
        
        Args:
            starting_index (int): The initial starting index for the sequential block.
            params (np.ndarray): The global array holding all parameters.
            grads (np.ndarray): The global array holding all gradients.
        
        Returns:
            int: The final index after all modules have been mapped.
        """
        current_index = starting_index
        for module in self._modules:
            current_index = module._init_indexes(current_index, params, grads)
        return current_index
    
    def init_random_params(self, default_activation: ActivationFunction = ReLU()) -> None:
        """Initializes weights of the linear layers based on their subsequent activation function.
        
        Traverses the modules in reverse to identify which activation function immediately
        follows a Linear layer, applying the appropriate statistical initialization (He, Xavier,
        etc.).
        
        Args:
            default_activation (ActivationFunction): The default activation logic to use. if a.
                                                     linear layer is not immediately. followed.
                                                     by an activation. Defaults to ReLU().
        
        Returns:
            None: This method initializes parameter values in place and returns no value.
        
        Raises:
            NotImplementedError: If the sequential container encounters an unsupported module.
                                 type. during initialization.
        
        Example:
            >>> obj = Sequential(...)
            >>> result = obj.init_random_params(default_activation=...)
        """
        current_activation = default_activation
        for module in reversed(self._modules):
            if isinstance(module, ActivationFunction):
                current_activation = module
            
            elif isinstance(module, Linear):
                current_activation.init_weight_function(module)
                current_activation = default_activation
            
            
            else:
                raise NotImplementedError(f"parameter initialisation is not implemeted for module : {type(module)}")
    
    @override
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Passes the input sequentially through all modules in the container.
        
        If a 1D array is passed, it is automatically reshaped into a 2D array (1, N) to ensure
        compatibility with matrix multiplications.
        
        Args:
            input (np.ndarray): The input data tensor.
        
        Returns:
            np.ndarray: The final output tensor after all transformations.
        
        Example:
            >>> obj = Sequential(...)
            >>> result = obj.forward(input=...)
        """
        # On mémorise si l'entrée était 1D avant de la modifier
        was_1d = (input.ndim == 1)
        
        if was_1d:
            input = input.reshape(1, -1)
            
        output = input
        for module in self._modules:
            output = module.forward(output)
        
        # Si on est entré avec un vecteur 1D, on ressort avec un vecteur 1D
        if was_1d:
            output = output.flatten() 
        
        return output
    
    @override
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Propagates the gradient backwards sequentially through all modules.
        
        The implementation uses the cached forward-pass data to propagate derivatives toward the
        previous stage and to accumulate any parameter gradients stored by the object.
        
        Args:
            dZ (np.ndarray): The upstream gradient to propagate backwards.
        
        Returns:
            np.ndarray: The gradient with respect to the original input of the Sequential block.
        
        Example:
            >>> obj = Sequential(...)
            >>> result = obj.backward(dZ=...)
        """
        for module in reversed(self._modules):
            dZ = module.backward(dZ)
        return dZ
    
    def reset_grads(self):
        """Resets all gradients in the global gradient array to zero.
        
        This should typically be called before starting a new backward pass to prevent gradients
        from accumulating across iterations.
        
        Returns:
            None: This method resets stored gradients in place and returns no value.
        
        Example:
            >>> obj = Sequential(...)
            >>> result = obj.reset_grads()
        """
        self._grads.fill(0)
    
    def params_copy(self, new_params: np.ndarray):
        """Safely updates the model's parameters by performing an in-place memory copy.
        
        This guarantees that internal layer references pointing to the global parameter array
        are preserved while the underlying values are updated (useful for applying optimizer
        steps).
        
        Args:
            new_params (np.ndarray): The new parameter array containing the values to copy.
        
        Returns:
            None: This method copies parameter values in place and returns no value.
        
        Raises:
            ValueError: If the shape of the new parameters does not match the current.
                        parameters.
        
        Example:
            >>> obj = Sequential(...)
            >>> result = obj.params_copy(new_params=...)
        """
        if new_params.shape != self._params.shape:
            raise ValueError("param_copy.shape != self._params.shape !!! Can't copy")
        else:
            self._params[:] = new_params



def main():
    nn = Sequential([
        Linear(12,34),
        Tanh(),
        Linear(34,4),
        ReLU(),
        Linear(4,12)
    ])
    
    print(nn)

if __name__ == "__main__":
    main()