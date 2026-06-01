"""
Main logic for saving and loading a network optimizer state.
"""
from __future__ import annotations
from .network_parser import Reader, Writer, FileFlags, SectionID
from ..core.nn import (
    NNModule,
    Sigmoid,
    ELU,
    Tanh,
    ReLU,
    LeakyReLU,
    Linear,
    Sequential)
from ..core.optim import (
    GradientOptimizer,
    SGD,
    Adam, 
    AdamW)
from ..rl.policy import (
    CriticNN,
    QNN,
    ActorNN)
from .parsing_functions import (
    read_nnmodule,
    read_normalizer,
    read_optimizer,
    read_hyperparams,
    write_nnmodule,
    write_normalizer,
    write_optimizer,
    write_hyperparams,

    read_sac_rollout,
    write_sac_rollout,
    )


from ..rl.utils import Normalizer, NoNormalizer, RunningMeanStd
from ..rl.base_nn_optimiser import NetworkOptimiser
from ..rl.openaies import OpenAIES
from ..rl.ppoptimiser import PPOptimiser, RolloutBuffer
from ..rl.sac_optimiser import SACOptimiser, SACRolloutBuffer
import gymnasium as gym
from pathlib import Path
from abc import abstractmethod, ABC
from typing import override

from dataclasses import dataclass

OPTIM_FILE_EXTENSION = ".st"


@dataclass
class NetworkOptimizerState(ABC):
    # For children of this class, multiple attributes are present (etc: actor, critic, obs_normalizer)
    
    @classmethod
    @abstractmethod
    # the arguments are just to give you an idea, they can be anything really
    def from_network_optimiser(cls, network_optimiser:NetworkOptimiser, obs_normalizer:Normalizer, hyperparams:NetworkOptimiser.HyperParams):
        """Auto-generate a state from a network optimiser"""
        ...

    @classmethod
    @abstractmethod
    def create(cls, reader:Reader) -> NetworkOptimizerState:
        """Read the network state from a file"""
        ...

    @abstractmethod
    def store(self, writer:Writer) -> None:
        """Store the inner networks, hyperparams, etc. to the file"""
        ...

@dataclass
class OpenAIESState(NetworkOptimizerState):
    main_network: Sequential
    optimizer: GradientOptimizer
    obs_normalizer: Normalizer
    hyper_params: OpenAIES.HyperParams
    # def __init__(self, optim:OpenAIES, normaliser:Normalizer | None, hyperparams:OpenAIES.HyperParams | None):
    #     self.network_optimizer:OpenAIES
    #     self.hyper_params:OpenAIES.HyperParams
    #     super().__init__(optim, normaliser if normaliser is not None else optim.obs_normalizer, hyperparams if hyperparams is not None else optim.hyper_params)


    @classmethod
    def from_network_optimiser(cls:type[OpenAIESState], network_optimiser:OpenAIES, obs_normalizer:Normalizer | None, hyperparams:OpenAIES.HyperParams | None):
        return cls(main_network=network_optimiser.main_network,
                   optimizer=network_optimiser.optimizer,
                   obs_normalizer=obs_normalizer if obs_normalizer is not None else network_optimiser.obs_normalizer, 
                   hyper_params=hyperparams if hyperparams is not None else network_optimiser.hyper_params)
        
    @override
    @classmethod
    def create(cls:type[OpenAIESState], reader:Reader):
        hp = read_hyperparams(reader)
        norm = read_normalizer(reader)
        network = read_nnmodule(reader)
        optim = read_optimizer(reader)
        return cls(main_network=network,
                   optimizer=optim,
                   obs_normalizer=norm,
                   hyper_params=hp)
        
    @override
    def store(self, writer:Writer):
        write_hyperparams(self.hyper_params, writer)
        write_normalizer(self.obs_normalizer, writer)
        write_nnmodule(self.main_network, writer)
        write_optimizer(self.optimizer, writer)

@dataclass
class PPOState(NetworkOptimizerState):
    actor: ActorNN
    critic: CriticNN
    actor_optimizer: GradientOptimizer
    critic_optimizer: GradientOptimizer
    out_of_action_optimizer: GradientOptimizer | None
    obs_normalizer: Normalizer
    hyper_params: PPOptimiser.HyperParams

    @classmethod
    def from_network_optimiser(cls:type[PPOState], network_optimiser:PPOptimiser, obs_normalizer:Normalizer | None, hyperparams:PPOptimiser.HyperParams | None):
        return cls(actor=network_optimiser.actor,
                   critic=network_optimiser.critic,
                   actor_optimizer=network_optimiser.actor_optimizer,
                   critic_optimizer=network_optimiser.critic_optimizer,
                   out_of_action_optimizer=network_optimiser.out_of_action_optimizer,
                   obs_normalizer=obs_normalizer if obs_normalizer is not None else network_optimiser.obs_normalizer, 
                   hyper_params=hyperparams if hyperparams is not None else network_optimiser.hyper_params)

    @override
    @classmethod
    def create(cls:type[PPOState], reader:Reader):
        hp = read_hyperparams(reader)
        norm = read_normalizer(reader)

        actor = read_nnmodule(reader)
        actor_optim = read_optimizer(reader)

        critic = read_nnmodule(reader)
        critic_optim = read_optimizer(reader)
        
        out_of_action_optim = read_optimizer(reader)
        if isinstance(out_of_action_optim, SGD) :
            if out_of_action_optim.alpha == None: # this is a dummy optimizer, meaning the original one was None
                out_of_action_optim = None

        return cls(actor=actor,
                   critic=critic,
                   actor_optimizer=actor_optim,
                   critic_optimizer=critic_optim,
                   out_of_action_optimizer=out_of_action_optim,
                   obs_normalizer=norm,
                   hyper_params=hp)

    @override
    def store(self, writer:Writer):
        # 1. write hyperparams
        write_hyperparams(self.hyper_params, writer)
        # 2. write observation normaliser
        write_normalizer(self.obs_normalizer, writer)
        # 3. write actor module and optimizer
        write_nnmodule(self.actor, writer)
        write_optimizer(self.actor_optimizer, writer)
        # 4. write critic module and optimizer
        write_nnmodule(self.critic, writer)
        write_optimizer(self.critic_optimizer, writer)
        # 5. write out of action optimizer if it exists
        write_optimizer(self.out_of_action_optimizer if self.out_of_action_optimizer is not None else SGD(maximise=True, alpha=None), writer) # dummy optimizer, won't be used if the original one is None
        

@dataclass
class SACState(NetworkOptimizerState):
    pi: ActorNN
    qs: tuple[QNN, QNN]
    q_tarrs: tuple[QNN, QNN]
    pi_optimizer: GradientOptimizer
    q_optimizers: tuple[GradientOptimizer, GradientOptimizer]
    alpha_optimizer: GradientOptimizer
    obs_normalizer: Normalizer
    hyper_params: SACOptimiser.HyperParams
    rollout_buffer: SACRolloutBuffer

    @classmethod
    def from_network_optimiser(cls:type[SACState], network_optimiser:SACOptimiser, obs_normalizer:Normalizer | None, hyperparams:SACOptimiser.HyperParams | None):
        return cls(pi=network_optimiser.pi,
                   qs=network_optimiser.qs,
                   q_tarrs=network_optimiser.q_tarrs,
                   pi_optimizer=network_optimiser.pi_optimizer,
                   q_optimizers=network_optimiser.q_optimizers,
                   alpha_optimizer=network_optimiser.alpha_optimizer,
                   obs_normalizer=obs_normalizer if obs_normalizer is not None else network_optimiser.obs_normalizer,
                   hyper_params=hyperparams if hyperparams is not None else network_optimiser.hyper_params,
                   rollout_buffer=network_optimiser.rollout_buffer)

    @override
    @classmethod
    def create(cls:type[SACState], reader:Reader):
        hp = read_hyperparams(reader)
        norm = read_normalizer(reader)
        buffer = read_sac_rollout(reader)

        actor = read_nnmodule(reader)
        actor_optim = read_optimizer(reader)

        qs = [None,None]
        q_tarrs = [None, None]
        q_optim = [None, None]
        qs[0] = read_nnmodule(reader)
        q_tarrs[0] = read_nnmodule(reader)
        q_optim[0] = read_optimizer(reader)

        qs[1] = read_nnmodule(reader)
        q_tarrs[1] = read_nnmodule(reader)
        q_optim[1] = read_optimizer(reader)
        alpha_optim = read_optimizer(reader)

        return cls(pi=actor,
                   qs=qs,
                   q_tarrs=q_tarrs,
                   pi_optimizer=actor_optim,
                   q_optimizers=q_optim,
                   alpha_optimizer=alpha_optim,
                   obs_normalizer=norm,
                   hyper_params=hp,
                   rollout_buffer=buffer)

    @override
    def store(self, writer:Writer):
        # 1. write hyperparams
        write_hyperparams(self.hyper_params, writer)
        # 2. write observation normaliser
        write_normalizer(self.obs_normalizer, writer)
        # 3. write the rollout buffer
        write_sac_rollout(self.rollout_buffer, writer)
        # 4. write the pi (actor) network
        write_nnmodule(self.pi, writer)
        write_optimizer(self.pi_optimizer, writer)
        # 5. Write the qs and q_tarrs twins
        write_nnmodule(self.qs[0], writer)
        write_nnmodule(self.q_tarrs[0], writer)
        write_optimizer(self.q_optimizers[0], writer)

        write_nnmodule(self.qs[1], writer)
        write_nnmodule(self.q_tarrs[1], writer)
        write_optimizer(self.q_optimizers[1], writer)
        
        # 6. write the alpha optimizer
        write_optimizer(self.alpha_optimizer, writer)



class CheckPoint:
    def __init__(self, metadata:str, environment:gym.Env | str, step:int, state:NetworkOptimizerState):
        """A checkpoint of the neural network optimiser's state at some point. Can be loaded from file or saved to file.
        
        Allows storing any arbitrary metadata, stores the environment's name (gym.Env.spec.id), the current epoch state, and the network optimizer's state (structure and tensors)

        Args:
            metadata (str): metadata to be carried by the file
            environment (gym.Env): environment. It will internally be converted to an environment id, so use gym.make() to get the environment
            step (int): the current epoch step
            state (NetworkOptimizerState): the network optimizer state to store
        """
        self.metadata:str = metadata
        if isinstance(environment, gym.Env):
            self.env_id:str = environment.spec.id
        elif isinstance(environment, gym.vector.VectorEnv):
            self.env_id:str = environment.spec.id
        elif isinstance(environment, str):
            self.env_id = environment
        else:
            raise TypeError("nuh uh")

        self.step:int = step
        self.state:NetworkOptimizerState = state

    def save(self, path:str | Path):
        writer = Writer(path)
        if self.step != 0:
            writer.flags |= FileFlags.INITIALIZED
        writer.step = self.step
        writer.env = self.env_id
        writer.algorithm = type(self.state).__name__
        writer.metadata = self.metadata
        # header section writes the step, flags and version
        writer.write_header_section()
        # we would normally write the metadata here (auto-begins and ends its section)
        writer.write_metadata_section()

        # now we go to the network section to write its preamble and initialize states
        writer.begin_section(SectionID.NETWORK)
        # use the state object to write its state to the file
        self.state.store(writer)
        # end the section (jumps back to beginning to write total size, then goes back to end)
        writer.end_section(SectionID.NETWORK)
        # all the arrays that have been referenced in the writing process need to be written to their own section
        writer.write_tensor_section()
        writer.file.close()


    @classmethod
    def load(cls:type[CheckPoint], path:str | Path) -> CheckPoint:
        reader = Reader(path)
        reader.parse_header()
        env = reader.env
        step = reader.step
        # TODO a better map of network optimizer -> optimizer state (to support mismatches or name changes)
        algos:dict[str, NetworkOptimizerState] = {
            OpenAIESState.__name__ : OpenAIESState,
            PPOState.__name__ : PPOState,
            SACState.__name__ : SACState
        }
        network = algos[reader.algorithm]
        reader.parse_metadata()
        meta = reader.metadata
        # we need to parse this section before actually loading the network optimizer, because it expects the read_tensor(tensor_id:int) function to work.
        # If we don't have our lookup table of arrays yet, it won't work and will beautifully crash horribly
        reader.parse_tensor()

        reader.goto_section(SectionID.NETWORK)
        state = network.create(reader)

        reader.file.close()

        return cls(meta, env, step, state)
        




"""
env = gym.make(env_name, kwargs)
net = PPOptimiser(
    env = env,
    networks = (actor, critic),
    actor_optim = actor_optim,
    critic_optim = critic_optim)
normaliser = RunningMeanStd(...)

hp = (...)

# train once (hyperparams are now bound)

state = PPOState(net, normaliser, hp=None) # hp can be infered because we already trained
cp = CheckPoint(metadata = "",
    environment = env,
    state = state
    )
cp.save("path/to/file")

--------------------------------------

loaded_point = CheckPoint.load("path/to/file")
assert type(loaded_point.state) == PPOState

env = gym.make(loaded_point.envname, loaded_point.metadata)
state = loaded_point.state
net = PPOptimiser(
    env = env,
    networks = (state.actor, state.critic),
    actor_optim = state.actor_optim,
    critic_optim = state.critic_optim)
normaliser = state.normaliser

hp = state.hyperparams





"""