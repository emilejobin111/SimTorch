"""Loss functions submodule for model evaluation.

This module provides a collection of loss functions used to measure the 
error between predicted values and ground truth data. It defines a standard 
interface for all loss functions to ensure consistent evaluation and 
gradient computation across different implementations.

Classes:
    Loss: An abstract base class defining the standard loss function interface.
    MSE: An implementation of the Mean Squared Error loss function.
"""


import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    """Abstract base class for loss functions.
    
    This class defines the standard interface for all loss functions, 
    requiring an evaluation method and a gradient computation method.
    """

    def __init__(self):
        """Initializes the Loss base class."""
        pass

    @abstractmethod
    def eval(self, predicted: np.ndarray, true_value: np.ndarray) -> float:
        """Evaluates the loss between predicted and true values.

        Args:
            predicted (np.ndarray): An array of predicted values.
            true_value (np.ndarray): An array of ground truth values.

        Returns:
            float: The calculated loss value.
        """
        pass

    @abstractmethod
    def grad(self) -> np.ndarray:
        """Computes the gradient of the loss with respect to the predictions.

        Returns:
            np.ndarray: An array containing the gradients.
        """
        pass

class MSE(Loss):
    """Mean Squared Error (MSE) loss function.
    
    Computes the mean squared error between the predictions and the ground truth.
    Stores the most recent inputs to be used for gradient computation.

    Attributes:
        predicted (np.ndarray): The most recently evaluated predicted values.
        true_value (np.ndarray): The most recently evaluated ground truth values.
    """

    def __init__(self):
        """Initializes the MSE loss function."""
        self.predicted: np.ndarray = None
        self.true_value: np.ndarray = None

    def eval(self, predicted: np.ndarray, true_value: np.ndarray) -> float:
        """Evaluates the Mean Squared Error (MSE).

        Args:
            predicted (np.ndarray): An array of predicted values.
            true_value (np.ndarray): An array of ground truth values.

        Returns:
            float: The computed mean squared error.
        """
        self.predicted = predicted
        self.true_value = true_value
        return np.mean((predicted - true_value)**2)

    def grad(self) -> np.ndarray:
        """Computes the gradient of the MSE loss with respect to the predictions.

        Returns:
            np.ndarray: The gradient array, having the same shape as `predicted`.
        """
        num_elements = self.true_value.size
        return 2 * (self.predicted - self.true_value) / num_elements