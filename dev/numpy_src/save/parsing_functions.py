from ..core.nn import (
    NNModule,
    Sigmoid,
    ELU,
    Tanh,
    ReLU,
    LeakyReLU,
    Linear,
    Sequential,
    HardShrink)
from ..core.optim import (
    GradientOptimizer,
    SGD,
    Adam, 
    AdamW)
from ..rl.policy import (
    CriticNN,
    QNN,
    ActorNN)
from ..rl.utils import Normalizer, NoNormalizer, RunningMeanStd
from ..rl.base_nn_optimiser import NetworkOptimiser
from ..rl.openaies import OpenAIES
from ..rl.ppoptimiser import PPOptimiser, RolloutBuffer
from ..rl.sac_optimiser import SACOptimiser, SACRolloutBuffer

from typing import TYPE_CHECKING, Callable, override, Generic, Protocol, TypeVar
if TYPE_CHECKING:
    from .network_parser import Writer, Reader
else:
    class Writer:...
    class Reader:...
import numpy as np

ID_MODULE:dict[int, NNModule] = {
    0 : Sigmoid,
    1 : ELU,
    2 : Tanh,
    3 : ReLU,
    4 : LeakyReLU,
    5 : Linear,
    6 : Sequential,
    10 : CriticNN,
    11 : QNN,
    12 : ActorNN,
    13 : HardShrink
}
MODULE_ID = {v: k for k, v in ID_MODULE.items()}

ID_GRADIENT:dict[int, GradientOptimizer] = {
    0 : SGD,
    1 : Adam,
    2 : AdamW
}
GRADIENT_ID = {v: k for k, v in ID_GRADIENT.items()}

ID_NORMALIZER:dict[int, Normalizer] = {
    0 : NoNormalizer,
    1 : RunningMeanStd
}
NORMALIZER_ID = {v: k for k, v in ID_NORMALIZER.items()}

ID_HYPERPARAMS:dict[int, NetworkOptimiser.HyperParams] = {
    0 : OpenAIES.HyperParams,
    1 : PPOptimiser.HyperParams,
    2 : SACOptimiser.HyperParams
}
HYPERPARAMS_ID = {v: k for k, v in ID_HYPERPARAMS.items()}




MODULE_MAP:dict[NNModule, tuple[Callable[[Reader],NNModule],Callable[[NNModule,Writer],None]]] = dict()
GRADIENT_MAP:dict[GradientOptimizer, tuple[Callable[[Reader],GradientOptimizer],Callable[[GradientOptimizer,Writer],None]]] = dict()
NORMALIZER_MAP:dict[Normalizer, tuple[Callable[[Reader],Normalizer],Callable[[Normalizer,Writer],None]]] = dict()
HYPERPARAMS_MAP:dict[NetworkOptimiser.HyperParams, tuple[Callable[[Reader],NetworkOptimiser.HyperParams],Callable[[NetworkOptimiser.HyperParams,Writer],None]]] = dict()

def register_read(type:NNModule|GradientOptimizer|Normalizer):
    def wrapper(func):
        if issubclass(type, NNModule):
            if MODULE_MAP.get(type) is None: MODULE_MAP[type] = [None, None]
            MODULE_MAP[type][0] = func
        elif issubclass(type, GradientOptimizer):
            if GRADIENT_MAP.get(type) is None: GRADIENT_MAP[type] = [None, None]
            GRADIENT_MAP[type][0] = func
        elif issubclass(type, Normalizer):
            if NORMALIZER_MAP.get(type) is None: NORMALIZER_MAP[type] = [None, None]
            NORMALIZER_MAP[type][0] = func
        elif issubclass(type, NetworkOptimiser.HyperParams):
            if HYPERPARAMS_MAP.get(type) is None: HYPERPARAMS_MAP[type] = [None,None]
            HYPERPARAMS_MAP[type][0] = func
        else:
            raise Exception(f"Cannot register a read function for type {type}")
        return func
    return wrapper

def register_write(type:NNModule|GradientOptimizer|Normalizer):
    def wrapper(func):
        if issubclass(type, NNModule):
            if MODULE_MAP.get(type) is None: MODULE_MAP[type] = [None, None]
            MODULE_MAP[type][1] = func
        elif issubclass(type, GradientOptimizer):
            if GRADIENT_MAP.get(type) is None: GRADIENT_MAP[type] = [None, None]
            GRADIENT_MAP[type][1] = func
        elif issubclass(type, Normalizer):
            if NORMALIZER_MAP.get(type) is None: NORMALIZER_MAP[type] = [None, None]
            NORMALIZER_MAP[type][1] = func
        elif issubclass(type, NetworkOptimiser.HyperParams):
            if HYPERPARAMS_MAP.get(type) is None: HYPERPARAMS_MAP[type] = [None,None]
            HYPERPARAMS_MAP[type][1] = func
        else:
            raise Exception(f"Cannot register a write function for type {type}")
        return func
    return wrapper

# Reading functions
# These functions read the first 4 bytes to know the layer id (or hyperparam type, or gradient optimizer, etc...)
def read_nnmodule(reader:Reader) -> NNModule:
    dtype = reader.read_u32()
    module_type = ID_MODULE.get(dtype)
    if module_type is None:
        raise KeyError(f"Unknown Module type of id {dtype}")
    return MODULE_MAP[module_type][0](reader)

def read_optimizer(reader:Reader) -> GradientOptimizer:
    dtype = reader.read_u32()
    grad_type = ID_GRADIENT.get(dtype)
    if grad_type is None:
        raise KeyError(f"Unknown Gradient optimizer type of id {dtype}")
    return GRADIENT_MAP[grad_type][0](reader)

def read_normalizer(reader:Reader) -> Normalizer:
    dtype = reader.read_u32()
    norm_type = ID_NORMALIZER.get(dtype)
    if norm_type is None:
        raise KeyError(f"Unknown observation normalizer type of id {dtype}")
    return NORMALIZER_MAP[norm_type][0](reader)

def read_hyperparams(reader:Reader) -> NetworkOptimiser.HyperParams:
    dtype = reader.read_u32()
    hp_type = ID_HYPERPARAMS.get(dtype)
    if hp_type is None:
        raise KeyError(f"Unknown hyperparams type of id {dtype}")
    return HYPERPARAMS_MAP[hp_type][0](reader)



def write_nnmodule(module: NNModule, writer:Writer) -> None:
    """Writes the module type to buffer then calls its handler to write internal data"""

    handlers = MODULE_MAP.get(type(module))
    if handlers is None or handlers[1] is None:
        raise KeyError(f"No known handler for Module {type(module)}")
    writer.write_u32(MODULE_ID[type(module)])
    return handlers[1](module, writer)

def write_optimizer(optim: GradientOptimizer, writer:Writer) -> None:
    """Writes the optimizer type to buffer then calls its handler to write internal data"""
    handlers = GRADIENT_MAP.get(type(optim))
    if handlers is None or handlers[1] is None:
        raise KeyError(f"No known handler for Gradient Optimizer {optim}")
    writer.write_u32(GRADIENT_ID[type(optim)])
    return handlers[1](optim, writer)

def write_normalizer(norm:Normalizer, writer:Writer) -> None:
    """Writes the normalizer type to buffer then calls its handler to write internal data"""
    handlers = NORMALIZER_MAP.get(type(norm))
    if handlers is None or handlers[1] is None:
        raise KeyError(f"No known handler for Normalizer {norm}")
    writer.write_u32(NORMALIZER_ID[type(norm)])
    return handlers[1](norm, writer)

def write_hyperparams(hp:NetworkOptimiser.HyperParams, writer:Writer) -> None:
    """Writes the normalizer type to buffer then calls its handler to write internal data"""
    x = type(hp)
    handlers = HYPERPARAMS_MAP.get(type(hp))
    if handlers is None or handlers[1] is None:
        raise KeyError(f"No known handler for hyperparams {hp}")
    writer.write_u32(HYPERPARAMS_ID[type(hp)])
    return handlers[1](hp, writer)
    

def read_sequential_like(reader:Reader) -> list[NNModule]:
    """Reads a list of NNModules in series and returns it. Uses the first 4 bytes to determine amount to read"""
    count = reader.read_u32()
    modules:list[NNModule] = []
    for _ in range(count):
        modules.append(read_nnmodule(reader))
    return modules

def write_sequential_like(modules:list[NNModule], writer:Writer):
    """Writes a list of NNModules in series. Encodes the length in the first 4 bytes"""
    writer.write_u32(len(modules))
    for module in modules:
        write_nnmodule(module, writer)



# MODULE READS
@register_read(Sequential)
def read_sequential(reader:Reader) -> Sequential:
    modules = read_sequential_like(reader)
    # count = reader.read_u32()
    # modules = []
    # for i in range(count):
    #     modules.append(read_nnmodule(reader))
    cls = Sequential(modules, default_init=True)
    params = reader.read_tensor()
    cls.params_copy(params)
    return cls

@register_write(Sequential)
def write_sequential(module: Sequential, writer:Writer):
    write_sequential_like(module._modules, writer)
    # count = len(module._modules)
    # writer.write_u32(count)
    # for module in module._modules:
    #     write_nnmodule(module, writer)
    # this is the main network weights being written
    writer.register_tensor(module.params)



@register_read(Linear)
def read_linear(reader:Reader) -> Linear:
    input_count = reader.read_u32()
    output_count = reader.read_u32()
    return Linear(input_count, output_count)

@register_write(Linear)
def write_linear(module:Linear, writer:Writer):
    writer.write_u32(module.input_count)
    writer.write_u32(module.output_count)



@register_read(Sigmoid)
def read_sigmoid(reader:Reader) -> Sigmoid:
    return Sigmoid()

@register_write(Sigmoid)
def write_sigmoid(module:Sigmoid, writer:Writer):
    pass



@register_read(ELU)
def read_ELU(reader:Reader) -> ELU:
    alpha = reader.read_f64()
    cls = ELU(alpha)
    return cls

@register_write(ELU)
def write_ELU(module:ELU, writer:Writer):
    writer.write_f64(module.alpha)



@register_write(Tanh)
def write_Tanh(*_:tuple[Tanh, Writer]):
    pass

@register_read(Tanh)
def read_Tanh(reader:Reader):
    return Tanh()



@register_write(ReLU)
def write_ReLU(module:ReLU, writer:Writer):
    pass

@register_read(ReLU)
def read_ReLU(reader:Reader):
    return ReLU()



@register_write(LeakyReLU)
def write_LeakyReLU(module:LeakyReLU, writer:Writer):
    writer.write_f64(module.alpha)

@register_read(LeakyReLU)
def read_LeakyReLU(reader:Reader):
    return LeakyReLU(reader.read_f64())



@register_write(HardShrink)
def write_HardSink(module:HardShrink, writer:Writer):
    writer.write_f64(module.lambda_)

@register_read(HardShrink)
def read_HardSink(reader:Reader):
    return HardShrink(reader.read_f64())



@register_read(ActorNN)
def read_ActorNN(reader:Reader) -> ActorNN:
    modules = read_sequential_like(reader)
    std = reader.read_bool()
    params = reader.read_tensor()
    cls = ActorNN(modules, std, True)
    cls.params_copy(params)
    return cls

@register_write(ActorNN)
def write_ActorNN(module:ActorNN, writer:Writer):
    write_sequential_like(module._modules, writer)
    writer.write_bool(module.standalone_std)
    writer.register_tensor(module.params)



@register_write(CriticNN)
def write_criticNNNNNN(module:CriticNN, writer:Writer):
    write_sequential_like(module._modules, writer)
    writer.register_tensor(module._params)

@register_read(CriticNN)
def read_criticNNNNNNN(reader:Reader):
    modules = read_sequential_like(reader)
    params = reader.read_tensor()
    cls = CriticNN(modules)
    cls.params_copy(params)
    return cls



@register_write(QNN)
def write_QNN(module:QNN, writer:Writer):
    write_criticNNNNNN(module, writer)

@register_read(QNN)
def read_QNN(reader:Reader):
    modules = read_sequential_like(reader)
    params = reader.read_tensor()
    cls = QNN(modules)
    cls.params_copy(params)
    return cls



@register_read(Adam)
def read_Adam(reader:Reader) -> Adam:
    cls = Adam(
        maximise=reader.read_bool(), # maximise
        alpha=reader.read_f64(),  # alpha
        beta1=reader.read_f64(),  # beta1
        beta2=reader.read_f64(),  # beta2
        epsilon=reader.read_f64())  # epsilon
    if reader.initialized:
        cls.initialized = True
        cls.m = reader.read_tensor()
        cls.v = reader.read_tensor()
        cls.t = reader.read_i64()
        cls.__m_hat = np.zeros_like(cls.m)
        cls.__v_hat = np.zeros_like(cls.m)

    return cls

@register_write(Adam)
def write_adam(grad:Adam, writer:Writer):
    writer.write_bool(grad.maximise)
    writer.write_f64(grad.alpha)
    writer.write_f64(grad.beta1)
    writer.write_f64(grad.beta2)
    writer.write_f64(grad.epsilon)
    if writer.initialized and grad.initialized:
        writer.register_tensor(grad.m)
        writer.register_tensor(grad.v)
        writer.write_i64(grad.t)



@register_read(AdamW)
def read_AdamW(reader:Reader) -> AdamW:
    cls = AdamW(
        maximise=reader.read_bool(),
        alpha=reader.read_f64(),
        weight_decay=reader.read_f64(),
        grad_clip=reader.read_f64(),
        beta1=reader.read_f64(),
        beta2=reader.read_f64(),
        epsilon=reader.read_f64())
    if reader.initialized:
        cls.initialized = True
        cls.weight_mask = reader.read_tensor()
        cls.m = reader.read_tensor()
        cls.v = reader.read_tensor()
        cls.t = reader.read_i64()
        cls.m_hat = np.zeros_like(cls.m)
        cls.v_hat = np.zeros_like(cls.m)

    return cls

@register_write(AdamW)
def write_adamW(grad:AdamW, writer:Writer):
    writer.write_bool(grad.maximise)
    writer.write_f64(grad.alpha)
    writer.write_f64(grad.weight_decay)
    writer.write_f64(grad.grad_clip)
    writer.write_f64(grad.beta1)
    writer.write_f64(grad.beta2)
    writer.write_f64(grad.epsilon)
    if writer.initialized and grad.initialized:
        writer.register_tensor(grad.weight_mask)
        writer.register_tensor(grad.m)
        writer.register_tensor(grad.v)
        writer.write_i64(grad.t)



@register_write(SGD)
def write_SGD(grad:SGD, writer:Writer):
    writer.write_bool(grad.maximise)
    writer.write_f64(grad.alpha)

@register_read(SGD)
def read_SGD(reader:Reader):
    cls = SGD(
        maximise=reader.read_bool(),
        alpha=reader.read_f64()
    )
    return cls



# NORMALIZZZERS
@register_write(NoNormalizer)
def write_nonormalizer(norm:NoNormalizer, writer:Writer):
    pass

@register_read(NoNormalizer)
def read_nonormalizer(reader:Reader):
    return NoNormalizer()



@register_write(RunningMeanStd)
def write_RunningMeanStd(norm:RunningMeanStd, writer:Writer):
    writer.register_tensor(norm.normalized_obs) # boolean mask
    writer.write_u64(norm.M2.shape[0]) # same as norm.__data_length but its private :(
    writer.register_tensor(norm.M2)
    writer.register_tensor(norm.mean)
    writer.write_u64(norm.count)

@register_read(RunningMeanStd)
def read_RunningMeanStd(reader:Reader):
    normalized_obs = reader.read_tensor()
    data_length = reader.read_u64()
    m2 = reader.read_tensor()
    mean = reader.read_tensor()
    count = reader.read_u64()
    cls = RunningMeanStd(data_length, normalized_obs)
    cls.M2 = m2
    cls.mean = mean
    cls.count = count
    return cls





# HyperParams
@register_write(OpenAIES.HyperParams)
def write_aieshp(hp:OpenAIES.HyperParams, writer:Writer):
    writer.write_u64(hp.num_mutant)
    writer.write_u64(hp.num_episodes)
    writer.write_u64(hp.start_epoch)
    writer.write_u64(hp.end_epoch)
    writer.write_f64(hp.sigma)
    writer.write_bool(hp.mirrored)
    writer.write_bool(hp.normalize_obs)

@register_read(OpenAIES.HyperParams)
def read_aieshp(reader:Reader) -> OpenAIES.HyperParams:
    hp = OpenAIES.HyperParams(
        num_mutant=reader.read_u64(),
        num_episodes=reader.read_u64(),
        start_epoch=reader.read_u64(),
        end_epoch=reader.read_u64(),
        sigma=reader.read_f64(),
        mirrored=reader.read_bool(),
        normalize_obs=reader.read_bool()
    )
    return hp



@register_write(PPOptimiser.HyperParams)
def write_ppohp(hp:PPOptimiser.HyperParams, writer:Writer):
    writer.write_u64(hp.end_epoch)
    writer.write_u64(hp.start_epoch)
    writer.write_u64(hp.rollout_length)
    writer.write_f64(hp.gamma)
    writer.write_f64(hp.lambd)
    writer.write_u64(hp.ppo_epoch)
    writer.write_u64(hp.mini_batch_size)
    writer.write_f64(hp.epsilon)
    writer.write_f64(hp.entropy_coef)
    writer.write_f64(hp.initial_best)
    writer.write_bool(hp.normalize_obs)
    writer.write_bool(hp.penalize_out_of_action)
    writer.write_f64(hp.out_of_action_acceptable_violation_rate)
    writer.write_f64(hp.out_of_action_learning_rate)
    writer.write_f64(float(hp.log_out_of_action_coeff) if hp.log_out_of_action_coeff != None else None)

@register_read(PPOptimiser.HyperParams)
def read_ppohp(reader:Reader) -> PPOptimiser.HyperParams:
    hp = PPOptimiser.HyperParams(
        end_epoch       = reader.read_u64(),
        start_epoch     = reader.read_u64(),
        rollout_length  = reader.read_u64(),
        gamma           = reader.read_f64(),
        lambd           = reader.read_f64(),
        ppo_epoch       = reader.read_u64(),
        mini_batch_size = reader.read_u64(),
        epsilon         = reader.read_f64(),
        entropy_coef    = reader.read_f64(),
        initial_best    = reader.read_f64(),
        normalize_obs   = reader.read_bool(),
        penalize_out_of_action = reader.read_bool(),
        out_of_action_acceptable_violation_rate = reader.read_f64(),
        out_of_action_learning_rate = reader.read_f64(),
        log_out_of_action_coeff = x if (x:=reader.read_f64()) == None else np.array(x, dtype=np.float32)
    )
    return hp

@register_write(SACOptimiser.HyperParams)
def write_sachp(hp:SACOptimiser.HyperParams, writer:Writer):
    writer.write_u64(hp.end_epoch)
    writer.write_u64(hp.start_epoch)
    writer.write_u64(hp.initial_exploration_length)
    writer.write_u64(hp.rollout_size)
    writer.write_u64(hp.step_amount)
    writer.write_u64(hp.batch_size)
    writer.write_f64(hp.gamma)
    # you can encode None into a NaN float
    writer.write_f64(float(hp.log_alpha) if hp.log_alpha != None else None)
    writer.write_f64(hp.tau)
    writer.write_f64(hp.tarr_entropy)
    writer.write_bool(hp.trainable_alpha)
    writer.write_bool(hp.normalize_obs)
    writer.write_u64(hp.test_network_at_epoch)
    writer.write_f64(hp.reward_scale)

@register_read(SACOptimiser.HyperParams)
def read_sachp(reader:Reader) -> SACOptimiser.HyperParams:
    hp = SACOptimiser.HyperParams(
        end_epoch = reader.read_u64(),
        start_epoch = reader.read_u64(),
        initial_exploration_length = reader.read_u64(),
        rollout_size = reader.read_u64(),
        step_amount = reader.read_u64(),
        batch_size = reader.read_u64(),
        gamma = reader.read_f64(),
        log_alpha = x if (x:=reader.read_f64()) == None else np.array(x, dtype=np.float32),
        tau = reader.read_f64(),
        tarr_entropy = reader.read_f64(),
        trainable_alpha = reader.read_bool(),
        normalize_obs = reader.read_bool(),
        test_network_at_epoch = reader.read_u64(),
        reward_scale = reader.read_f64()
    )
    return hp



# Custom read write for SAC

def read_sac_rollout(reader:Reader) -> SACRolloutBuffer:
    rollout_size = reader.read_u64()
    index = reader.read_u64()
    buffer = SACRolloutBuffer(0,0, rollout_size)
    buffer.index = index
    buffer.obs = reader.read_tensor()
    buffer.actions = reader.read_tensor()
    buffer.next_obs = reader.read_tensor()
    buffer.rewards = reader.read_tensor()
    buffer.dones = reader.read_tensor()
    return buffer

def write_sac_rollout(buffer:SACRolloutBuffer, writer:Writer):
    writer.write_u64(buffer.rollout_size)
    writer.write_u64(buffer.index)
    writer.register_tensor(buffer.obs)
    writer.register_tensor(buffer.actions)
    writer.register_tensor(buffer.next_obs)
    writer.register_tensor(buffer.rewards)
    writer.register_tensor(buffer.dones)




if __name__ == "__main__":


    for one_map in (MODULE_MAP, GRADIENT_MAP, GRADIENT_MAP, HYPERPARAMS_MAP):
        for key, value in one_map.items():
            if value[0] is None:
                print(f"{key} is missing a read function")
            if value[1] is None:
                print(f"{key} is missing a write function")
