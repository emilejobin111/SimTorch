"""
A network file configuration contains all data required to setup an environment and whatnot from a packed file

Structure:
[Header]
- filetype
- version of the file

[Metadata]
    [Training]
    - algorithm
    - step

    [Environment]
    - name
    - data to create an env

[Network]
- observation normalizer
- for each network:
    - Structure (binary nnmodule description)
    - (Gradient) Optimiser

[Tensors]
- for each module
    - raw vector data
- for each gradient optimizer
    - parameters to initialize


"""

"""
Saving:
1. Knowing Network optimiser, do something idk
2. Store gym env name and its data so we can remake it
3. Store network HyperParams (dict) (serilalize custom for sac), used for training
4. Store observation normalizer
OPENAIES:
    1. store network (sequential)
    2. store gradient optimizer
    3. store total steps
PPOPTIMISER:
    1. store critic network (sequential)
    2. store actor network (sequential)

"""


"""
Parsing:
"raw" data is stored as [dtype]{data}.
For floats, ints, etc. this is very simple, the dtype tells us the length to read

However, if we have a list (array), the data is going to be different. That data is the list length,
and we need to read the next field that amount of times. The next fields will be in [dtype]{data} structure.
It may look something like this [list]{3}[float]{2.1}[int]{2}[bool]{false} (all in binary)
You can have lists inside lists, because the reading will use recursion

For a module, the data tells us another id : the module id. [dtype][moduletype][data].
After knowing the module type, the data after is the same as before, but without the dtype field
before each value. That's because knowing the module tells us the data layout.


"""
from dataclasses import dataclass

from ..core.nn import NNModule, Sequential
from ..core.optim import GradientOptimizer
from ..rl.base_nn_optimiser import NetworkOptimiser
from ..rl.utils import Normalizer, RunningMeanStd, NoNormalizer
from ..rl.openaies import OpenAIES
from ..rl.ppoptimiser import PPOptimiser
from ..rl.sac_optimiser import SACOptimiser

import numpy as np





class OptimizerSerializer:
    @staticmethod
    def serialize(optim : NetworkOptimiser) -> bytes:
        if isinstance(optim, OpenAIES):
            ...
        elif isinstance(optim, PPOptimiser):
            ...
        elif isinstance(optim, SACOptimiser):
            ...

    @staticmethod
    def load(data:bytes) -> NetworkOptimiser:
        ...


@dataclass
class NetworkConfig:
    """Description of one network"""
    structure:NNModule | Sequential # or actor / critic, serialize it with the same system
    optim: GradientOptimizer
    has_twin: bool  # SAC has some networks that are doubled, so we need to create two networks from one config
    total_size: int # length of the tensor so we can calculate offsets in the file (double it and split in the middle if has twin)

@dataclass
class NetworkOptimizerConfig:
    """Description of the whole network we can train"""
    hyper_params: NetworkOptimiser.HyperParams
    obs_normaliser: Normalizer
    networks: tuple[NetworkConfig, ...] # one or multiple configs (1 for openai, 2 for ppo and 3 for SAC)

@dataclass
class FileStructure:
    version:int

    algorithm:str # ppo, sac or openaies
    current_step:int # if 0, untrained network with no tensors
    
    environment:str
    env_data:tuple   # should be able to do gym.make with the name and env data

    network: NetworkOptimizerConfig

    tensors: tuple[np.ndarray] # all packed in contiguous data. We use individual network lengths to offset our reads of these arrays



class NetworkFile:
    def __init__(self):
        ...

    def metadata(self) -> dict:
        ...

    def network(self) -> dict:
        ...