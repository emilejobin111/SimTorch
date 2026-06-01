"""
The module defines lightweight metadata descriptors and routines for reading and writing
model checkpoints, normalizers, and optimizer state.
"""
import struct
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..rl.utils import NoNormalizer, RunningMeanStd

from ..core.nn import NNModule, Linear, Tanh, ReLU, LeakyReLU, ELU, Sequential
from ..core.optim import GradientOptimizer
import struct

from ..core import optim

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "autodrone" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_checkpoint_name(training_name: str) -> str:
    """Convert a user-facing training name into a filesystem-safe folder name."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", training_name.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("training_name must contain at least one alphanumeric character")
    return sanitized


def get_checkpoint_dir(training_name: str | None = None) -> Path:
    """Return the checkpoint directory for a training run, creating it if needed."""
    if training_name is None:
        return CHECKPOINT_DIR

    checkpoint_dir = CHECKPOINT_DIR / sanitize_checkpoint_name(training_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

@dataclass
class ParamDesc:
    """Descriptor that records metadata for param serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Attributes:

        dtype (type): Value supplied for `dtype`.

        name (str): Value supplied for `name`.

        attr (str): Value supplied for `attr`.

        default (Any): Value supplied for `default`.

    

    Example:

        >>> obj = ParamDesc()

    """



    dtype:type
    name:str
    attr: str
    default:Any = None
class ModuleDesc:
    """Descriptor that records metadata for module serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Attributes:

        descs (list[ParamDesc]): Value supplied for `descs`.

        optim (ClassVar[type[GradientOptimizer]]): Value supplied for `optim`.

    

    Example:

        >>> obj = ModuleDesc()

    """



    descs: list[ParamDesc] = []
    optim: ClassVar[type[GradientOptimizer]]
    module: NNModule

class OptimDesc:
    """Descriptor that records metadata for optim serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Attributes:

        descs (list[ParamDesc]): Value supplied for `descs`.

        optim (GradientOptimizer): Value supplied for `optim`.

    

    Example:

        >>> obj = OptimDesc()

    """



    descs: list[ParamDesc] = []
    optim: GradientOptimizer = None

class LinearDesc(ModuleDesc):
    """Descriptor that records metadata for linear serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Example:

        >>> obj = LinearDesc()

    """



    descs = [
        ParamDesc(int, "input count",  attr="_input_count"),
        ParamDesc(int, "output count", attr="_output_count"),
    ]
    module = Linear
class TanhDesc(ModuleDesc):
    """Descriptor that records metadata for tanh serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Example:

        >>> obj = TanhDesc()

    """



    descs = []
    module = Tanh
class ReLUDesc(ModuleDesc):
    """Descriptor that records metadata for re lu serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Example:

        >>> obj = ReLUDesc()

    """



    descs = []
    module = ReLU
class LeakyReluDesc(ModuleDesc):
    """Descriptor that records metadata for leaky relu serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Example:

        >>> obj = LeakyReluDesc()

    """



    descs = [
        ParamDesc(float, "alpha", attr="_LeakyReLU__alpha", default=0.01)
    ]
    module = LeakyReLU
class ELUDesc(ModuleDesc):
    """Descriptor that records metadata for elu serialization.

    

    These descriptor objects are consumed by the save and load helpers when reconstructing

    modules or optimizer state from compact binary data.

    

    Example:

        >>> obj = ELUDesc()

    """



    descs = [
        ParamDesc(float, "alpha", attr="_ELU__alpha", default=1.0)
    ]
    module = ELU

TAGS:dict[int, ModuleDesc] = {
    1:LinearDesc,
    2:LeakyReluDesc,
    3:TanhDesc,
    4:ReLUDesc,
    5:ELUDesc
}
    
MAP:dict[NNModule, ModuleDesc] = {
    Linear : LinearDesc,
    LeakyReLU : LeakyReluDesc,
    Tanh : TanhDesc,
    ReLU : ReLUDesc,
    ELU : ELUDesc
}


#####

"""
    int     -> 4 bytes <I
    float32 -> 4 bytes <f
    float64 -> 8 bytes <d
    uint64  -> 8 bytes <Q
    bool    -> 1 byte  <B
"""
# ╭────────────────────────────────────────────────────────╮
# │                   Strucutre&metadata                   │
# ╰────────────────────────────────────────────────────────╯

def read_metadata(path:str) -> tuple[Sequential, float, np.ndarray, NoNormalizer|RunningMeanStd, bool, int, int, int]:
    """Read a checkpoint file and reconstruct its saved contents.
    
    The routine participates in the checkpointing pipeline used to persist or reconstruct
    SimTorch models and related runtime state.
    
    Args:
        path (str): Filesystem path or path prefix used for persistence.
    
    Returns:
        tuple[...]:
        network, score, normalizer, trainning_status (((( 0=never trained; 1=trainning; 2= trainned; 3=unknown))))),
        model_type ((((((((0=POO; 1=SAC; 2=OpenAIes; 3= unknown)))),
        epoch_count (int)
    
    Raises:
        ValueError: Raised if one or more argument values are invalid or unsupported.
    
    Example:
        >>> result = read_metadata(path="checkpoint")
    """
    with open(path, "rb") as f:
        # MÉTADONNÉES
        score = struct.unpack("<d", f.read(8))[0]
        epoch_count = struct.unpack("<I", f.read(4))[0]
        training_status: int = struct.unpack("<I", f.read(4))[0] # 0=never trained; 1=trainning; 2= trainned; 3=unknown 
        model_type = struct.unpack("<I", f.read(4))[0]

        # NORMALISEUR
        norm_tag = struct.unpack("<B", f.read(1))[0]
        normalizer = NoNormalizer()      
        if norm_tag == 1:
            L     = struct.unpack("<Q", f.read(8))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            mean  = np.frombuffer(f.read(L * 4), dtype=np.float32).copy()
            m2    = np.frombuffer(f.read(L * 4), dtype=np.float32).copy()

            normalizer = RunningMeanStd(data_length=L)
            normalizer.load_state({"mean": mean, "M2": m2, "count": count})
        elif norm_tag != 0:
            raise ValueError(f"Tag de normaliseur inconnu : {norm_tag}")



        # STRUCTURE
        modules = []
        while True: # i am evil and this is not ai
            tag = struct.unpack("<I", f.read(4))[0]

            if tag == 0xFF:
                break
            
            values = []
            for param_desc in TAGS[tag].descs:
                if param_desc.dtype == int:
                    values.append(struct.unpack("<I", f.read(4))[0])
                elif param_desc.dtype == float:
                    values.append(struct.unpack("<f", f.read(4))[0])
                else:
                    raise ValueError(f"Unsupported dtype {param_desc.dtype}")

            modules.append(TAGS[tag].module(*values))

    network = Sequential(modules)

    return network, score, normalizer, training_status, model_type, epoch_count


def write_metadata(path:str, network:Sequential, score:float, normalizer:NoNormalizer|RunningMeanStd, trainning_status:int=-1, model_type:int=3, epoch_count:int=0):
    """Write a checkpoint file containing network metadata and parameters.
    
    The routine participates in the checkpointing pipeline used to persist or reconstruct
    SimTorch models and related runtime state.
    
    Args:
        path (str): Filesystem path or path prefix used for persistence.
        network (Sequential): Neural network instance used by the routine.
        score (float): Current score or metric used by the routine.
        normalizer (NoNormalizer | RunningMeanStd): Observation normalizer to save, load,.
                                                    or. apply.
        trainning_status (int): 0=never trained; 1=trainning; 2= trainned; 3=unknown
        model_type (int): 0=POO; 1=SAC; 2=OpenAIes; 3= unknown
        epoch_count (int): Epoch index associated with the saved training state.
    
    Returns:
        None: This function writes the checkpoint contents and returns no value.
    
    Raises:
        ValueError: Raised if one or more argument values are invalid or unsupported.
    
    Example:
        >>> result = write_metadata(path="checkpoint", network=..., score=0.1, ...)
    """
    with open(path, "wb") as f:
        # MÉTADONNÉES
        f.write(struct.pack("<d", score))
        f.write(struct.pack("<I", epoch_count))
        f.write(struct.pack("<I", trainning_status))# 0=never trained; 1=trainning; 2= trainned; 3=unknown
        f.write(struct.pack("<I", model_type)) # 0=POO; 1=SAC; 2=OpenAIes; 3= unknown


        # NORMALISEUR
        if isinstance(normalizer, NoNormalizer):
            f.write(struct.pack("<B", 0))
        elif isinstance(normalizer, RunningMeanStd):
            f.write(struct.pack("<B", 1))
            f.write(struct.pack("<Q", len(normalizer.mean)))
            f.write(struct.pack("<Q", normalizer.count))
            f.write(normalizer.mean.astype(np.float32).tobytes())
            f.write(normalizer.M2.astype(np.float32).tobytes())
        else:
            raise ValueError(f"Unsupported normalizer type: {type(normalizer)}")

        # STRUCTURE
        for module in network._modules:
            # unwrap a nested Sequential (e.g. produced by enable_done)
            inner_modules = module._modules if isinstance(module, Sequential) else [module]
            for m in inner_modules:
                desc = MAP[type(m)]
                tag = list(TAGS.keys())[list(TAGS.values()).index(desc)]
                f.write(struct.pack("<I", tag))

                for param_desc in desc.descs:
                    value = getattr(m, param_desc.attr)
                    if param_desc.dtype == int:
                        f.write(struct.pack("<I", value))
                    elif param_desc.dtype == float:
                        f.write(struct.pack("<f", value))
                    else:
                        raise ValueError(f"Unsupported dtype {param_desc.dtype}")
        
        # Marqueur de fin de structure
        f.write(struct.pack("<I", 0xFF))
        print(f"Sauvegarde → {path}")


# ╭────────────────────────────────────────────────────────╮
# │                         params                         │
# ╰────────────────────────────────────────────────────────╯

def save_params(path: str, params: np.ndarray) -> None:
    """Saves the network parameters into a true NumPy .npz archive.
    
    Args:
        path (str): Filesystem path or path prefix.
        params (np.ndarray): Shared parameter buffer that stores trainable values.
    """
    saved_path = str(path) + ".npz"
    np.savez(saved_path, params=params)
    print(f"Sauvegarde des paramètres → {saved_path}")

def load_params(path: str) -> np.ndarray:
    """Loads parameters from a true NumPy .npz archive."""
        
    # Load the archive
    data = np.load(path)
    
    # Extract the specific 'params' array
    params = data["params"]
    print(f"Chargement des paramètres → {path}")
    
    return params


# ╭────────────────────────────────────────────────────────╮
# │                       optimizer                        │
# ╰────────────────────────────────────────────────────────╯

def save_optimizer_states(
    actor_optim: optim.GradientOptimizer,
    critic_optim: optim.GradientOptimizer,
    path: str,
    epoch: int,
    hyper_params,        
) -> None:
    """Saves actor and critic optimizer states to <path>_optim.npz.
    
    The routine participates in the checkpointing pipeline used to persist or reconstruct
    SimTorch models and related runtime state.
    
    Args:
        actor_optim (optim.GradientOptimizer): Optimizer state container for the actor.
                                               network.
        critic_optim (optim.GradientOptimizer): Optimizer state container for the critic.
                                                network.
        path (str): Filesystem path or path prefix used for persistence.
        epoch (int): Epoch index associated with the saved training state.
        hyper_params (Any): Hyperparameter object controlling the training loop.
    
    Returns:
        None: This function persists optimizer state and returns no value.
    
    Example:
        >>> result = save_optimizer_states(actor_optim=..., critic_optim=..., path="checkpoint", ...)
    """
    arrays = {}
    
    def add_optimizer_state(prefix: str, optimizer_state) -> None:
        if optimizer_state is None:
            return
        if isinstance(optimizer_state, (list, tuple)):
            for index, nested_optim in enumerate(optimizer_state):
                add_optimizer_state(f"{prefix}{index}", nested_optim)
            return

        state = optimizer_state.state()
        for key, val in state.items():
            arrays[f"{prefix}_{key}"] = np.array(val)

    add_optimizer_state("actor", actor_optim)
    add_optimizer_state("critic", critic_optim)

    arrays["epoch"] = np.array(epoch)

    
    if dataclasses.is_dataclass(hyper_params) and not isinstance(hyper_params, type):
        for hp_field in dataclasses.fields(hyper_params):
            arrays[f"hp_{hp_field.name}"] = np.array(getattr(hyper_params, hp_field.name))

    save_path = path + "_optim.npz"
    np.savez(save_path, **arrays)
    print(f"Sauvegarde → {save_path}")

def load_optimizer_states(
    actor_optim  : optim.GradientOptimizer,
    critic_optim : optim.GradientOptimizer,
    path         : str,
    hyper_params_cls,# type[PPOptimiser.HyperParams]
) -> tuple[int, object]:
    """Restores optimizer states from <path>_optim.npz.
    
    The routine participates in the checkpointing pipeline used to persist or reconstruct
    SimTorch models and related runtime state.
    
    Args:
        actor_optim (optim.GradientOptimizer): Optimizer state container for the actor.
                                               network.
        critic_optim (optim.GradientOptimizer): Optimizer state container for the critic.
                                                network.
        path (str): Filesystem path or path prefix used for persistence.
        hyper_params_cls (Any): Class used to reconstruct the saved hyperparameter object.
    
    Returns:
        tuple[int, object]: The epoch to resume from (saved epoch + 1).
    
    Example:
        >>> result = load_optimizer_states(actor_optim=..., critic_optim=..., path="checkpoint", ...)
    """
    data = np.load(path + "_optim.npz")

    actor_state  = {k.removeprefix("actor_"):  data[k] for k in data if k.startswith("actor_")}
    critic_state = {k.removeprefix("critic_"): data[k] for k in data if k.startswith("critic_")}

    if actor_state:
        actor_optim.load_state(actor_state)
    if critic_state:
        critic_optim.load_state(critic_state)

    start_epoch = int(data["epoch"]) + 1
 
    # Reconstruct hyperparameters
    hp_fields = { 
        hp_field.name: hp_field 
        for hp_field in dataclasses.fields(hyper_params_cls)
    }

    kwargs = {}
    for key in data:
        if key.startswith("hp_"):
            name = key.removeprefix("hp_")
            if name in hp_fields:
                raw = data[key]
                field_type = hp_fields[name].type
                # Cast back to the right Python type
                if field_type in (int, "int"):
                    kwargs[name] = int(raw)
                elif field_type in (float, "float"):
                    kwargs[name] = float(raw)
                elif field_type in (bool, "bool"):
                    kwargs[name] = bool(raw)
                else:
                    kwargs[name] = raw.item() if hasattr(raw, "item") else raw
 
    kwargs["start_epoch"] = start_epoch
    hyper_params = hyper_params_cls(**kwargs)
 
    print(f"Chargement → {path}_optim.npz")
    return start_epoch, hyper_params
