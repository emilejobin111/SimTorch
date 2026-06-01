"""Actor, critic, and Q-network helpers for reinforcement learning.

The module wraps the core sequential networks with actor-critic-specific constraints and
convenience builders used by the PPO and SAC pipelines.
"""

from __future__ import annotations
from typing import Tuple, override, Optional
from copy import deepcopy

import numpy as np

from ..core import nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
class CriticNN(nn.Sequential):
    """A neural network acting as the Critic in an Actor-Critic architecture.
    
    This network evaluates the quality of a state. It is strictly constrained to output a
    single scalar value representing the estimated state value (V).
    
    Args:
        modules (list[nn.NNModule]): An ordered list of neural network modules.
    
    Example:
        >>> obj = CriticNN(modules=...)
    """


    def __init__(self, modules: list[nn.NNModule]):
        """Initialize a new CriticNN instance.
        
        Args:
            modules (list[nn.NNModule]): Ordered modules used to build a sequential network.
        """
        super().__init__(modules)
        self.__last_output_compatibility_check()
        
    @classmethod
    def build_critic(cls:CriticNN,
                     obs_dim: int,
                     hidden_layers_count: int = 2,
                     hidden_dims: int = 256,
                     activation_func: Optional[nn.ActivationFunction]=None)->CriticNN:
        
        """Build and return a critic instance.
        
        The helper returns ready-to-use objects so calling code can construct common
        architectures with a single call.
        
        Args:
            obs_dim (int): Observation-space dimensionality used to size the network.
            hidden_layers_count (int): Number of hidden layers to create.
            hidden_dims (int): Width assigned to each generated hidden layer.
            activation_func (Optional[nn.ActivationFunction]): Activation module copied between.
                                                               generated hidden layers.
        
        Returns:
            CriticNN: Newly constructed objects matching the requested configuration.
        
        Example:
            >>> result = CriticNN.build_critic(obs_dim=4, hidden_layers_count=4, hidden_dims=4, ...)
        """
        if activation_func is None:
            activation_func = nn.ReLU()
        modules = []
        
        if hidden_layers_count == 0:
            modules.append(nn.Linear(obs_dim,1))
            modules.append(deepcopy(activation_func))
            return cls(modules)
        
        modules.append(nn.Linear(obs_dim,hidden_dims))
        modules.append(deepcopy(activation_func))
        
        for _ in range(hidden_layers_count - 1):
            modules.append(nn.Linear(hidden_dims,hidden_dims))
            modules.append(deepcopy(activation_func))
        
        modules.append(nn.Linear(hidden_dims,1))
        
        return cls(modules)
    
    
    @classmethod
    def from_sequential(cls, network: nn.Sequential):
        """Creates a CriticNN instance from an existing Sequential network.
        
        This class method deep copies the modules of the provided network and replaces its final
        trainable layer (e.g., a Linear layer) with a new layer of the same type that has
        exactly one output dimension. This ensures the resulting network is compatible with the
        Critic architecture.
        
        Args:
            network (nn.Sequential): The base sequential network to copy and adapt.
        
        Returns:
            Any: A new Critic neural network ready to output a single state value.
        
        Raises:
            nn.InitializationException: Raised if the component is used before initialization is.
                                     complete.
        
        Example:
            >>> result = CriticNN.from_sequential(network=...)
        """
        copy_modules = deepcopy(network._modules)
        
        for i in reversed(range(len(copy_modules))):
            if hasattr(copy_modules[i], "_output_count"):
                new_last = type(copy_modules[i])(input_count = copy_modules[i]._input_count, output_count=1)
                copy_modules[i] = new_last
                return cls(copy_modules)
        raise nn.InitializationException("there was no trainable module in the nn given")
    
    def __last_output_compatibility_check(self):
        """Verifies that the network configuration results in exactly one output feature.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Raises:
            nn.InitializationException: Raised if the component is used before initialization is.
                                     complete.
        
        Example:
            >>> obj = CriticNN(...)
            >>> result = obj.__last_output_compatibility_check()
        """
        if self.output_count > 1:
            raise nn.InitializationException("A Critic nn can only have one output")
        elif self.output_count == 1:
            return
        else:
            raise nn.InitializationException("A Critic nn must have at least one output")
        

class QNN(CriticNN):
    """A neural network acting as a Q-function approximator in an Actor-Critic architecture.
    
    This network evaluates the quality of a state-action pair. It is strictly constrained to
    output a single scalar value representing the estimated action value (Q). It inherits
    from CriticNN since both architectures share the same output structure and evaluation
    purpose, but QNNs are typically trained with different loss functions.
    
    Example:
        >>> obj = QNN()
    """


    @classmethod
    def build_Q(cls:QNN,
                obs_dim: int,
                act_dim: int,
                hidden_layers_count: int = 2,
                hidden_dims: int = 256,
                activation_func: Optional[nn.ActivationFunction]=None)->CriticNN:
        
        """Build and return a q instance.
        
        The helper returns ready-to-use objects so calling code can construct common
        architectures with a single call.
        
        Args:
            obs_dim (int): Observation-space dimensionality used to size the network.
            act_dim (int): Action-space dimensionality used to size the network.
            hidden_layers_count (int): Number of hidden layers to create.
            hidden_dims (int): Width assigned to each generated hidden layer.
            activation_func (Optional[nn.ActivationFunction]): Activation module copied between.
                                                               generated hidden layers.
        
        Returns:
            CriticNN: Newly constructed objects matching the requested configuration.
        
        Example:
            >>> result = QNN.build_Q(obs_dim=4, act_dim=4, hidden_layers_count=4, ...)
        """
        if activation_func is None:
            activation_func = nn.ReLU()
        
        modules = []
        
        if hidden_layers_count == 0:
            modules.append(nn.Linear(obs_dim+act_dim,1))
            modules.append(deepcopy(activation_func))
            return cls(modules)
        
        modules.append(nn.Linear(obs_dim+act_dim,hidden_dims))
        modules.append(deepcopy(activation_func))
        
        for _ in range(hidden_layers_count - 1):
            modules.append(nn.Linear(hidden_dims,hidden_dims))
            modules.append(deepcopy(activation_func))
        
        modules.append(nn.Linear(hidden_dims,1))
        
        return cls(modules)
    @classmethod
    def from_sequential_add_act(cls, network: nn.Sequential, act_dim: int):
        """Creates a QNN instance from an existing Sequential network."""

        copy_modules = deepcopy(network._modules)
        
        for i in range(len(copy_modules)):
            if hasattr(copy_modules[i], "_output_count"):
                new_first = type(copy_modules[i])(input_count = copy_modules[i]._input_count+act_dim, output_count=copy_modules[i]._output_count)
                copy_modules[i] = new_first
                return super().from_sequential(nn.Sequential(copy_modules))
            
        raise nn.InitializationException("there was no trainable module in the nn given")
    
    
    @override
    def forward(self, input):
        """Performs the forward pass for the Q-function approximator.
        
        This method calculates the estimated Q-value for a given state-action pair. If the input
        is a tuple or list, it is expected to contain two elements: the observation (state)
        array and the action array, in the order [observation, action]. These two components are
        concatenated along the last axis before passing them through the underlying sequential
        network. Otherwise, it passes the input directly to the parent's forward method.
        
        Args:
            input (Any): Input array passed to the computation.
        
        Returns:
            Any: A single scalar value representing the estimated state-action value Q(s, a).
                 The shape will typically be (batch_size, 1).
        
        Example:
            >>> obj = QNN(...)
            >>> result = obj.forward(input=...)
        """
        if isinstance(input,(tuple,list)):
            input = np.concatenate(input, axis=-1)
        return super().forward(input)

#TODO make a discrete actor
class ActorNN(nn.Sequential):
    """A neural network acting as the Actor in an Actor-Critic architecture.
    
    This network predicts continuous actions by outputting both a mean (mu) and a standard
    deviation (sigma) to define a normal distribution. It handles its own dynamic memory
    allocation to accommodate either state-dependent or state-independent standard
    deviations.
    
    Args:
        modules (list[nn.NNModule]): An ordered list of neural network modules.
        standalone_std (bool): If True, the standard deviation is a learned. global.
                               parameter independent of the input state. If False, the.
                               network's. last layer is expanded to predict the standard.
                               deviation dynamically. based on the state. Defaults to True.
    
    Example:
        >>> obj = ActorNN(modules=..., standalone_std=False)
    """


    def __init__(self, modules: list[nn.NNModule], standalone_std: bool = True, already_init: bool = False):
        
        # On empêche l'initialisation par défaut pour gérer la mémoire nous-mêmes
        """Initialize a new ActorNN instance.
        
        Args:
            modules (list[nn.NNModule]): Ordered modules used to build a sequential network.
            standalone_std (bool): Whether the actor keeps an input-independent standard.
                                   deviation parameter.
        """
        super().__init__(modules, default_init=False) 
        
        
        self.__standalone_std = standalone_std
        if not already_init:
            self.__action_dim = self.output_count
            if standalone_std:
                self.__add_standalone_std()
            else:
                self.__add_std_output()
        else:
            if standalone_std:
                self.__action_dim = self.output_count
                self._param_count += self.__action_dim
                
                self.__std_indexs = np.array([self._param_count - self.__action_dim, self._param_count])
            else:
                for i in reversed(range(len(self._modules))):
                    if hasattr(self._modules[i], "_output_count"):
                        if self._modules[i]._output_count % 2 != 0:
                            raise nn.InitializationException("The output dimension of the last layer must be divisible by 2 to separate mean and std outputs")
                        self.__std_indexs = np.array([self._modules[i]._output_count//2, self._modules[i]._output_count])
                        self.__action_dim = self._modules[i]._output_count//2
                        break
            self._params = np.zeros(shape=(self.param_count,), dtype=np.float32)
            self._grads = np.zeros(shape=(self.param_count,), dtype=np.float32)
            
            self._init_indexes(0, self._params, self._grads)
    
    @classmethod
    def build_actor(cls:ActorNN,
                    obs_dim: int,
                    act_dim: int,
                    hidden_layers_count: int = 2,
                    hidden_dims: int = 256,
                    activation_func: Optional[nn.ActivationFunction]=None,
                    standalone_std: bool = True)->ActorNN:
        
        """Build and return a actor instance.
        
        The helper returns ready-to-use objects so calling code can construct common
        architectures with a single call.
        
        Args:
            obs_dim (int): Observation-space dimensionality used to size the network.
            act_dim (int): Action-space dimensionality used to size the network.
            hidden_layers_count (int): Number of hidden layers to create.
            hidden_dims (int): Width assigned to each generated hidden layer.
            activation_func (Optional[nn.ActivationFunction]): Activation module copied between.
                                                               generated hidden layers.
            standalone_std (bool): Whether the actor keeps an input-independent standard.
                                   deviation parameter.
        
        Returns:
            CriticNN: Newly constructed objects matching the requested configuration.
        
        Example:
            >>> result = ActorNN.build_actor(obs_dim=4, act_dim=4, hidden_layers_count=4, ...)
        """
        if activation_func is None:
            activation_func = nn.ReLU()
        
        modules = []
        
        if hidden_layers_count == 0:
            modules.append(nn.Linear(obs_dim,act_dim))
            modules.append(deepcopy(activation_func))
            return cls(modules, standalone_std)
        
        modules.append(nn.Linear(obs_dim,hidden_dims))
        modules.append(deepcopy(activation_func))
        
        for _ in range(hidden_layers_count - 1):
            modules.append(nn.Linear(hidden_dims,hidden_dims))
            modules.append(deepcopy(activation_func))
        
        modules.append(nn.Linear(hidden_dims,act_dim))
        
        return cls(modules,standalone_std)
    
    
    @classmethod
    def from_sequential(cls, network: nn.Sequential, standalone_std: bool = True, already_init: bool = False):
        """Creates an ActorNN instance from an existing Sequential network.
        
        This class method deep copies the modules of the provided base network and initializes a
        new Actor network. It allows for the easy instantiation of an actor from a generic
        architectural template while configuring its standard deviation strategy.
        
        Args:
            network (nn.Sequential): The base sequential network to copy.
            standalone_std (bool): If True, uses a state-independent. standard deviation. If.
                                   False, the network's final layer will be. modified to.
                                   predict. the standard deviation dynamically. Defaults to.
                                   True.
        
        Returns:
            Any: A new Actor neural network ready for continuous action spaces.
        
        Example:
            >>> result = ActorNN.from_sequential(network=..., standalone_std=False)
        """
        module_copy = deepcopy(network._modules)
        actor = cls(module_copy, standalone_std)
        if already_init:
            actor.param_count(network.params)
        return actor
    
    
    def __add_standalone_std(self):
        """Allocates memory for a state-independent standard deviation.
        
        Appends a separate parameter vector at the end of the global memory array to store the
        raw log-standard deviation values.
        
        Returns:
            None: This method performs its work in place and returns no value.
        """
        base_param_count = self._param_count
        self._param_count += self.__action_dim
        
        self._params = np.zeros(shape=(self.param_count,), dtype=np.float32)
        self._grads = np.zeros(shape=(self.param_count,), dtype=np.float32)
        
        self._init_indexes(0, self._params, self._grads)
        
        self.__std_indexs = np.array([base_param_count, self.param_count])
        
    def __add_std_output(self):
        """Modifies the final Linear layer to output state-dependent standard deviations.
        
        Doubles the output dimension of the last Linear module to accommodate both the mean and
        the raw log-standard deviation for each action dimension.
        
        Returns:
            None: This method performs its work in place and returns no value.
        
        Raises:
            NotImplementedError: If the layer preceding the output is not a Linear module. or.
                                 an. ActivationFunction. nn.InitialisationException: If no.
                                 valid. module is found to append the std output.
            nn.InitializationException: Raised if the component is used before initialization is.
                                     complete.
        
        Example:
            >>> obj = ActorNN(...)
            >>> result = obj.__add_std_output()
        """
        for i in reversed(range(len(self._modules))):
            if isinstance(self._modules[i], nn.ActivationFunction):
                pass
            elif isinstance(self._modules[i], nn.Linear):
                self.__std_indexs = np.array([self.__action_dim, self.__action_dim * 2])
                self._modules[i] = nn.Linear(self._modules[i]._input_count, self.__action_dim * 2)
                
                param_count: int = 0
                for module in self._modules:
                    param_count += module.param_count
                
                self._param_count = param_count
                self._params = np.zeros(shape=(self.param_count,))
                self._grads = np.zeros(shape=(self.param_count,))
                
                self._init_indexes(0, self._params, self._grads)
                return
            else:
                raise NotImplementedError(f"The module type {type(self._modules[i])} is not supported for adding a std output")
        raise nn.InitializationException("No module found in the actor network to add the std output to")
        
    @property
    def standalone_std(self) -> bool:
        """Property to check if the network uses a state-independent standard deviation.
        
        Returns:
            bool: True if the standard deviation is standalone, False otherwise.
        """
        return self.__standalone_std
        
    @property
    def std(self) -> np.ndarray | None:
        """Property to get the current global standard deviation.
        
        Returns:
            np.ndarray | None: The mathematical standard deviation (exp(raw_std)) if using
                               standalone_std, otherwise returns None.
        """
        if self.__standalone_std:
            raw_std = self._params[self.__std_indexs[0]:self.__std_indexs[1]]
            return np.exp(raw_std) # to return the real std
        else:
            return None
    
    def deterministic_forward(self, input: np.ndarray) -> np.ndarray:
        """Performs a forward pass returning only the mean action.
        
        Typically used during evaluation or exploitation phases where exploration noise is not
        desired. Automatically handles both 1D and 2D inputs.
        
        Args:
            input (np.ndarray): The environment state or observation.
        
        Returns:
            np.ndarray: The predicted mean action (mu).
        
        Example:
            >>> obj = ActorNN(...)
            >>> result = obj.deterministic_forward(input=...)
        """
        is_1d = input.ndim == 1
        if is_1d:
            input = input.reshape(1, -1)
        output = self.forward(input)[:, :self.__action_dim]
        if is_1d:
            output = output.flatten()  
        return output
    
    def stochastic_forward(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a forward pass returning both the mean and standard deviation.
        
        Uses a Hard Clip combined with a Straight-Through Estimator (in the backward pass) to
        safely bound the log standard deviation between -20 and 2. This prevents the vanishing
        gradient problem caused by Tanh saturation.
        
        Args:
            input (np.ndarray): Input array passed to the computation.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Sampled actions and associated standard deviations.
        
        Example:
            >>> obj = ActorNN(...)
            >>> result = obj.stochastic_forward(input=...)
        """
        is_1d = input.ndim == 1
        if is_1d:
            input = input.reshape(1, -1)
            
        forward_pass = self.forward(input)
        mean = forward_pass[:, :self.__action_dim]
        
        if self.standalone_std:
            raw_std = self._params[self.__std_indexs[0]:self.__std_indexs[1]]
            raw_std = np.tile(raw_std, (mean.shape[0], 1))
        else:
            raw_std = forward_pass[:, self.__std_indexs[0]:self.__std_indexs[1]]
            
        #^ --- HARD CLIP LOGIC ---
        
        clipped_log_std = np.clip(raw_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = np.exp(clipped_log_std)
        self.__std_buffer = std
        #^ -----------------------
        
        if is_1d:
            mean = mean.flatten()
            std = std.flatten()
            
        return mean, std
    
    @override
    def backward(self, dmean: np.ndarray, dstd: np.ndarray) -> np.ndarray:
        """Performs the backward pass for the Actor network.
        
        Implements the Straight-Through Estimator (STE) for the standard deviation. The gradient
        ignores the forward clipping boundaries, ensuring the network can always recover from
        extreme determinism without vanishing gradients.
        
        Args:
            dmean (np.ndarray): Array containing dmean values.
            dstd (np.ndarray): Array containing dstd values.
        
        Returns:
            np.ndarray: Gradient propagated to the previous stage.
        
        Example:
            >>> obj = ActorNN(...)
            >>> result = obj.backward(dmean=..., dstd=...)
        """
        if dmean.ndim == 1:
            dmean = dmean.reshape(1, -1)
            dstd = dstd.reshape(1, -1)
            
        #^ --- Straight-Through Estimator Backward ---
        d_clipped_log_std = dstd * self.__std_buffer
        
        d_raw_std = d_clipped_log_std
        #^ -------------------------------------------
        
        if self.__standalone_std:
            self._grads[self.__std_indexs[0]:self.__std_indexs[1]] += np.sum(d_raw_std, axis=0)
            return super().backward(dmean)
        else:
            dZ = np.hstack((dmean, d_raw_std))
            return super().backward(dZ)
    
    @override
    def init_random_params(self, default_activation: nn.ActivationFunction = nn.Tanh(), std_value: float = 1) -> None:
        """Initializes the network weights and the standard deviation parameters.
        
        Overrides the base Sequential initialization to specifically handle the
        state-independent standard deviation array. The raw standard deviation parameters are
        initialized using the natural logarithm of the desired target standard deviation to
        ensure mathematical consistency when processed through the exponential function during
        the forward pass.
        
        Args:
            default_activation (nn.ActivationFunction): The default activation. logic to use.
                                                        for. linear layers not immediately.
                                                        followed. by an. activation function.
                                                        Defaults to. nn.Tanh().
            std_value (float): The target initial standard deviation (sigma). for the action.
                               distribution. Defaults to 1.
        
        Returns:
            None: This method initializes parameter values in place and returns no value.
        
        Example:
            >>> obj = ActorNN(...)
            >>> result = obj.init_random_params(default_activation=..., std_value=0.1)
        """
        if self.standalone_std:
            #? --- np.log() because we dont use true std but log(std) ---
            self._params[self.__std_indexs[0]:self.__std_indexs[1]] = np.log(std_value)
            #? ----------------------------------------------------------
        super().init_random_params(default_activation)


def make_network_from_sequential_for_openaies(network: nn.Sequential) -> nn.Sequential:
    """
    Creates a standalone copy of a sequential network for OpenAI Evolution Strategies.

    Since Evolution Strategies (ES) typically optimize a single policy by perturbing 
    parameters, this function provides a deep copy of the template to ensure 
    the original architecture remains untouched during the mutation process.

    Args:
        network (nn.Sequential): The template neural network architecture.

    Returns:
        nn.Sequential: A deep-copied instance of the input network.
    """
    net = deepcopy(network)
    return net


def make_networks_from_sequential_for_ppo(network: nn.Sequential,
                                         standalone_std: bool = True
                                         ) -> Tuple[ActorNN, CriticNN]:
    """
    Constructs Actor and Critic networks from a sequential template for PPO.

    Proximal Policy Optimization (PPO) requires an Actor to select actions and a 
    Critic to estimate value functions. This factory splits the template logic 
    into these two distinct components.

    Args:
        network (nn.Sequential): The template neural network architecture.
        standalone_std (bool): If True, the action standard deviation is treated 
            as a separate learnable parameter rather than a state-dependent output. 
            Defaults to True.

    Returns:
        Tuple[ActorNN, CriticNN]: A tuple containing the initialized (Actor, Critic).
    """
    actor = ActorNN.from_sequential(network, standalone_std)
    critic = CriticNN.from_sequential(network)
    nets = (actor, critic)
    return nets


def make_networks_from_sequential_for_sac(network: nn.Sequential, 
                                         act_dim: int, 
                                         standalone_std: bool = False
                                         ) -> Tuple[ActorNN, QNN, QNN]:
    """
    Initializes Actor and twin Q-networks for Soft Actor-Critic (SAC).

    To combat overestimation bias, SAC utilizes Clipped Double-Q Learning, 
    requiring two separate Q-networks. This function also sets up the stochastic 
    Actor policy.

    Args:
        network (nn.Sequential): The template neural network architecture.
        act_dim (int): The dimensionality of the action space, required for 
            the Q-network input.
        standalone_std (bool): Whether the Actor's standard deviation is 
            independent of the input state. Defaults to False.

    Returns:
        Tuple[ActorNN, QNN, QNN]: A tuple containing (Actor, Q1, Q2).
    """
    actor = ActorNN.from_sequential(network, standalone_std)
    q_network1 = QNN.from_sequential_add_act(network, act_dim)
    q_network2 = QNN.from_sequential_add_act(network, act_dim)
    nets = (actor, q_network1, q_network2)
    return nets



def make_actor_critic_tuple(obs_dim: int,
                            act_dim: int,
                            hidden_layers_count: int = 2,
                            hidden_dims: int = 256,
                            activation_func: Optional[nn.ActivationFunction]=None,
                            standalone_std: bool = True
                            )->Tuple[ActorNN, CriticNN]:
    
    """Construct and return actor critic tuple.
    
    The helper returns ready-to-use objects so calling code can construct common
    architectures with a single call.
    
    Args:
        obs_dim (int): Observation-space dimensionality used to size the network.
        act_dim (int): Action-space dimensionality used to size the network.
        hidden_layers_count (int): Number of hidden layers to create.
        hidden_dims (int): Width assigned to each generated hidden layer.
        activation_func (Optional[nn.ActivationFunction]): Activation module copied between.
                                                           generated hidden layers.
        standalone_std (bool): Whether the actor keeps an input-independent standard.
                               deviation parameter.
    
    Returns:
        Tuple[ActorNN, CriticNN]: Newly constructed objects matching the requested
                                  configuration.
    
    Example:
        >>> result = make_actor_critic_tuple(obs_dim=4, act_dim=4, hidden_layers_count=4, ...)
    """
    actor = ActorNN.build_actor(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func, standalone_std)
    critic = CriticNN.build_critic(obs_dim, hidden_layers_count, hidden_dims, activation_func)
    return actor, critic

def make_q_and_actor_tuple(obs_dim: int,
                           act_dim: int,
                           hidden_layers_count: int = 2,
                           hidden_dims: int = 256,
                           activation_func: Optional[nn.ActivationFunction]=None,
                           standalone_std: bool = True)->tuple[ActorNN, QNN, QNN]:
    
    """Construct and return q and actor tuple.
    
    The helper returns ready-to-use objects so calling code can construct common
    architectures with a single call.
    
    Args:
        obs_dim (int): Observation-space dimensionality used to size the network.
        act_dim (int): Action-space dimensionality used to size the network.
        hidden_layers_count (int): Number of hidden layers to create.
        hidden_dims (int): Width assigned to each generated hidden layer.
        activation_func (Optional[nn.ActivationFunction]): Activation module copied between.
                                                           generated hidden layers.
        standalone_std (bool): Whether the actor keeps an input-independent standard.
                               deviation parameter.
    
    Returns:
        Tuple[ActorNN, QNN, QNN]: Newly constructed objects matching the requested
                                  configuration.
    
    Example:
        >>> result = make_q_and_actor_tuple(obs_dim=4, act_dim=4, hidden_layers_count=4, ...)
    """
    actor = ActorNN.build_actor(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func, standalone_std)
    q_network1 = QNN.build_Q(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func)
    q_network2 = QNN.build_Q(obs_dim, act_dim, hidden_layers_count, hidden_dims, activation_func)
    return actor, q_network1, q_network2

def make_nn(obs_dim: int,
            act_dim: int,
            hidden_layers_count: int = 2,
            hidden_dims: int = 256,
            activation_func: Optional[nn.ActivationFunction]=None)->nn.Sequential:
    
    """Construct and return nn.
    
    The helper returns ready-to-use objects so calling code can construct common
    architectures with a single call.
    
    Args:
        obs_dim (int): Observation-space dimensionality used to size the network.
        act_dim (int): Action-space dimensionality used to size the network.
        hidden_layers_count (int): Number of hidden layers to create.
        hidden_dims (int): Width assigned to each generated hidden layer.
        activation_func (Optional[nn.ActivationFunction]): Activation module copied between.
                                                           generated hidden layers.
    
    Returns:
        nn.Sequential: Newly constructed objects matching the requested configuration.
    
    Example:
        >>> result = make_nn(obs_dim=4, act_dim=4, hidden_layers_count=4, ...)
    """
    if activation_func is None:
        activation_func = nn.ReLU()
    
    modules = []
    
    if hidden_layers_count == 0:
        modules.append(nn.Linear(obs_dim, act_dim))
        modules.append(deepcopy(activation_func))
        return nn.Sequential(modules)
    
    modules.append(nn.Linear(obs_dim, hidden_dims))
    modules.append(deepcopy(activation_func))
    
    for _ in range(hidden_layers_count - 1):
        modules.append(nn.Linear(hidden_dims, hidden_dims))
        modules.append(deepcopy(activation_func))
    
    modules.append(nn.Linear(hidden_dims, act_dim))
    
    return nn.Sequential(modules)