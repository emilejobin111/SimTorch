from __future__ import annotations
from ..core.scheduler import LinearScheduler
from ..core.optim import GradientOptimizer

from ..rl.openaies import OpenAIES
from ..rl.ppoptimiser import PPOptimiser
from ..rl.sac_optimiser import SACOptimiser
from ..rl.base_nn_optimiser import NetworkOptimiser
import gymnasium as gym
from pathlib import Path
from json import dumps, loads

from .checkpoint import CheckPoint, PPOState, OpenAIESState, SACState


def make_meta_data_from_scheduler(scheduler: LinearScheduler) -> dict:
    """Extracts the internal state of a LinearScheduler for serialization.
    
    Args:
        scheduler (LinearScheduler): The learning rate scheduler to serialize.
        
    Returns:
        dict: A dictionary containing the start value, end value, total steps, and current step (t).
        
    Example:
        >>> meta = make_meta_data_from_scheduler(my_scheduler)
    """
    meta_data = {
        "start_value": scheduler.start_value,
        "end_value": scheduler.end_value,
        "num_steps": scheduler.num_steps,
        "t": scheduler.t
    }
    return meta_data

def get_scheduler_from_meta_data(meta_data: dict, optimiser: GradientOptimizer) -> LinearScheduler | None:
    """Reconstructs a LinearScheduler from serialized metadata.
    
    Automatically binds the newly created scheduler to the 'alpha' (learning rate) 
    property of the provided optimizer.
    
    Args:
        meta_data (dict): The dictionary containing the serialized scheduler state.
        optimiser (GradientOptimizer): The optimizer the scheduler will control.
        
    Returns:
        LinearScheduler | None: The reconstructed scheduler, or None if the metadata is invalid or incomplete.
        
    Example:
        >>> scheduler = get_scheduler_from_meta_data(meta, my_optimiser)
    """
    start_value = meta_data.get("start_value", None)
    end_value = meta_data.get("end_value", None)
    num_steps = meta_data.get("num_steps", None)
    t = meta_data.get("t", None)
    if start_value is not None and end_value is not None and num_steps is not None and t is not None:
        scheduler = LinearScheduler(start_value=start_value,
                                    end_value=end_value,
                                    num_steps=num_steps,
                                    object_and_property_name=(optimiser, "alpha"))
        scheduler.t = t
        return scheduler
    else:
        return None

def load_checkpoint_to_optim(path: Path, num_envs: int = 1, async_vec_env: bool=True) -> tuple[NetworkOptimiser, dict|None]:
    checkpoint = CheckPoint.load(path)
    if isinstance(checkpoint.state,PPOState):
        return load_checkpoint_to_ppo(checkpoint, num_envs, async_vec_env)
    elif isinstance(checkpoint.state, OpenAIESState):
        return load_checkpoint_to_openaies(checkpoint)
    elif isinstance(checkpoint.state, SACState):
        return load_checkpoint_to_sac(checkpoint, num_envs, async_vec_env)
    else:
        raise ValueError("Unknown checkpoint type")

def save_checkpoint_from_ppo(optimiser: PPOptimiser,
                             path: Path,
                             step: int=1,
                             meta_data: dict=None) -> None:
    """Saves the complete state of a PPOptimiser to a binary checkpoint file.
    
    Serializes the neural networks, optimizers, observation normalizers, hyperparameters,
    schedulers, and any user-provided external metadata.
    
    Args:
        optimiser (PPOptimiser): The PPO instance to save.
        path (Path): The file path where the checkpoint will be written.
        step (int): The current training epoch or step. Defaults to 1.
        meta_data (dict, optional): Custom external metadata to store alongside the model. Defaults to None.
        
    Example:
        >>> save_checkpoint_from_ppo(ppo, Path("model.st"), step=50, meta_data={"env": "Ant-v5"})
    """
    meta_data = {"external_metadata": meta_data}
    internal_metadata = {}
    state = PPOState.from_network_optimiser(optimiser, None, None)
    schedulers = optimiser.schedulers
    actor_scheduler: LinearScheduler = schedulers.get("actor", None)
    critic_scheduler: LinearScheduler = schedulers.get("critic", None)
    if not isinstance(actor_scheduler, LinearScheduler):
        actor_scheduler = None
    else:
        internal_metadata["actor_scheduler"] = make_meta_data_from_scheduler(actor_scheduler)
    if not isinstance(critic_scheduler, LinearScheduler):
        critic_scheduler = None
    else:
        internal_metadata["critic_scheduler"] = make_meta_data_from_scheduler(critic_scheduler)
    meta_data["internal_metadata"] = internal_metadata
    meta_data_str = dumps(meta_data)
    checkpoint = CheckPoint(metadata=meta_data_str, environment=optimiser.env, step=step, state=state)
    checkpoint.save(path)

def load_checkpoint_to_ppo(checkpoint: Path|CheckPoint, num_envs: int, async_vec_env: bool = True) -> tuple[PPOptimiser, dict|None]:
    """Reconstructs a fully functional PPOptimiser from a saved checkpoint file.
    
    Rebuilds the gymnasium environment, restores network weights and optimizer momentum, 
    and re-attaches the learning rate schedulers.
    
    Args:
        path (Path): The file path of the saved checkpoint or the raw checkpoint object.
        num_envs (int): The number of parallel environments to instantiate for the vectorized env.
        
    Returns:
        tuple[PPOptimiser, dict|None]: A tuple containing the restored PPOptimiser instance 
                                       and the user's external metadata.
        
    Raises:
        ValueError: If the checkpoint cannot be loaded or is not of type PPOState.
        
    Example:
        >>> ppo, meta = load_checkpoint_to_ppo(Path("model.st"), num_envs=4)
    """
    if isinstance(checkpoint, (Path,str)):
        path = checkpoint
        checkpoint = CheckPoint.load(path)
        if checkpoint is None:
            raise ValueError(f"Failed to load checkpoint from {path}")
    state = checkpoint.state
    if not isinstance(state, PPOState):
        raise ValueError(f"Checkpoint state is not of type PPOState, got {type(state)}")
    meta_data_str = checkpoint.metadata
    meta_data: dict = loads(meta_data_str)
    internal_metadata:dict = meta_data.get("internal_metadata", {})
    external_metadata:dict = meta_data.get("external_metadata", {})
    if async_vec_env:
        env = gym.make_vec(checkpoint.env_id, num_envs=num_envs,vectorization_mode=gym.VectorizeMode.ASYNC)
    else:
        env = gym.make_vec(checkpoint.env_id, num_envs=num_envs)
    optimiser = PPOptimiser(env=env,
                            networks=(state.actor, state.critic),
                            actor_optimizer=state.actor_optimizer,
                            critic_optimizer=state.critic_optimizer)
    optimiser.setup_training(hyper_params=state.hyper_params,
                             obs_normalizer=state.obs_normalizer,
                             schedulers={})
    actor_scheduler_meta = internal_metadata.get("actor_scheduler", None)
    critic_scheduler_meta = internal_metadata.get("critic_scheduler", None)
    if actor_scheduler_meta is not None:
        actor_scheduler = get_scheduler_from_meta_data(actor_scheduler_meta, optimiser.actor_optimizer)
        if actor_scheduler is not None:
            optimiser.schedulers["actor"] = actor_scheduler
    if critic_scheduler_meta is not None:
        critic_scheduler = get_scheduler_from_meta_data(critic_scheduler_meta, optimiser.critic_optimizer)
        if critic_scheduler is not None:
            optimiser.schedulers["critic"] = critic_scheduler
    if state.hyper_params.penalize_out_of_action:
        optimiser.out_of_action_optimizer = state.out_of_action_optimizer
        optimiser.out_of_action_optimizer.initialize(state.hyper_params.log_out_of_action_coeff)
    
    state.hyper_params.start_epoch = checkpoint.step
    return optimiser, external_metadata


def save_checkpoint_from_openaies(optimiser: OpenAIES,
                                  path: Path,
                                  step: int=1,
                                  meta_data: dict=None) -> None:
    """Saves the complete state of an OpenAIES optimizer to a binary checkpoint file.
    
    Serializes the main neural network, optimizer, observation normalizers, hyperparameters,
    schedulers, and any user-provided external metadata.
    
    Args:
        optimiser (OpenAIES): The OpenAIES instance to save.
        path (Path): The file path where the checkpoint will be written.
        step (int): The current training epoch or step. Defaults to 1.
        meta_data (dict, optional): Custom external metadata to store alongside the model. Defaults to None.
        
    Example:
        >>> save_checkpoint_from_openaies(es, Path("es_model.st"), step=100)
    """
    meta_data_dict = {"external_metadata": meta_data}
    internal_metadata = {}
    
    state = OpenAIESState.from_network_optimiser(optimiser, None, None)
    schedulers = optimiser.schedulers
    
    main_scheduler: LinearScheduler = schedulers.get("main", schedulers.get("actor", None))
    
    if isinstance(main_scheduler, LinearScheduler):
        internal_metadata["main_scheduler"] = make_meta_data_from_scheduler(main_scheduler)
    
    meta_data_dict["internal_metadata"] = internal_metadata
    meta_data_str = dumps(meta_data_dict)
    
    checkpoint = CheckPoint(metadata=meta_data_str, environment=optimiser.env, step=step, state=state)
    checkpoint.save(path)


def load_checkpoint_to_openaies(checkpoint: Path|CheckPoint, num_envs: int = 1) -> tuple[OpenAIES, dict|None]: 
    """Reconstructs a fully functional OpenAIES optimizer from a saved checkpoint file.
    
    Rebuilds the environment, restores network weights, optimizer state, and learning rate schedulers.
    
    Args:
        path (Path): The file path of the saved checkpoint or the raw checkpoint object.
        num_envs (int): Number of environments (usually 1 for ES, kept for API consistency). Defaults to 1.
        
    Returns:
        tuple[OpenAIES, dict|None]: A tuple containing the restored OpenAIES instance 
                                    and the user's external metadata.
                                    
    Raises:
        ValueError: If the checkpoint cannot be loaded or is not of type OpenAIESState.
        
    Example:
        >>> es, meta = load_checkpoint_to_openaies(Path("es_model.st"))
    """
    if isinstance(checkpoint, (Path,str)):
        path = checkpoint
        checkpoint = CheckPoint.load(path)
        if checkpoint is None:
            raise ValueError(f"Failed to load checkpoint from {path}")
    if not isinstance(checkpoint, CheckPoint):
        raise ValueError(f"Checkpoint is not of type CheckPoint, got {type(checkpoint)}")
        
    state = checkpoint.state
    if not isinstance(state, OpenAIESState):
        raise ValueError(f"Checkpoint state is not of type OpenAIESState, got {type(state)}")
        
    meta_data_str = checkpoint.metadata
    meta_data: dict = loads(meta_data_str)
    
    internal_metadata: dict = meta_data.get("internal_metadata", {})
    external_metadata: dict = meta_data.get("external_metadata", {})
    
    env = gym.make(checkpoint.env_id)
    
    optimiser = OpenAIES(env=env,
                         network=state.main_network, 
                         optimizer=state.optimizer)
                         
    optimiser.setup_training(hyper_params=state.hyper_params,
                             obs_normalizer=state.obs_normalizer,
                             schedulers={})
                             
    main_scheduler_meta = internal_metadata.get("main_scheduler", internal_metadata.get("actor_scheduler"))
    
    if main_scheduler_meta is not None:
        main_scheduler = get_scheduler_from_meta_data(main_scheduler_meta, optimiser.optimizer)
        if main_scheduler is not None:
            optimiser.schedulers["main"] = main_scheduler
            
    state.hyper_params.start_epoch = checkpoint.step
    return optimiser, external_metadata


def save_checkpoint_from_sac(optimiser: SACOptimiser,
                             path: Path,
                             step: int=1,
                             meta_data: dict=None) -> None:
    """Saves the complete state of a SACOptimiser to a binary checkpoint file.
    
    Serializes the actor policy, the twin Q-networks (and their delayed targets), 
    the rollout buffer, the entropy temperature parameter (alpha), and all associated 
    optimizers and schedulers.
    
    Args:
        optimiser (SACOptimiser): The SAC instance to save.
        path (Path): The file path where the checkpoint will be written.
        step (int): The current training epoch or step. Defaults to 1.
        meta_data (dict, optional): Custom external metadata to store alongside the model. Defaults to None.
        
    Example:
        >>> save_checkpoint_from_sac(sac, Path("sac_model.st"), step=500)
    """
    if step != 1:
        optimiser.hyper_params.initial_exploration_length = 0 # We don't want to reset the exploration phase if we're resuming training from a checkpoint
    meta_data_dict = {"external_metadata": meta_data}
    internal_metadata = {}
    
    state = SACState.from_network_optimiser(optimiser, None, None)
    schedulers = optimiser.schedulers
    
    pi_scheduler: LinearScheduler = schedulers.get("pi", None)
    q1_scheduler: LinearScheduler = schedulers.get("q1", None)
    q2_scheduler: LinearScheduler = schedulers.get("q2", None)
    alpha_scheduler: LinearScheduler = schedulers.get("alpha", None)
    
    if isinstance(pi_scheduler, LinearScheduler):
        internal_metadata["pi_scheduler"] = make_meta_data_from_scheduler(pi_scheduler)
    if isinstance(q1_scheduler, LinearScheduler):
        internal_metadata["q1_scheduler"] = make_meta_data_from_scheduler(q1_scheduler)
    if isinstance(q2_scheduler, LinearScheduler):
        internal_metadata["q2_scheduler"] = make_meta_data_from_scheduler(q2_scheduler)
    if isinstance(alpha_scheduler, LinearScheduler):
        internal_metadata["alpha_scheduler"] = make_meta_data_from_scheduler(alpha_scheduler)
        
    meta_data_dict["internal_metadata"] = internal_metadata
    meta_data_str = dumps(meta_data_dict)
    
    checkpoint = CheckPoint(metadata=meta_data_str, environment=optimiser.env, step=step, state=state)
    checkpoint.save(path)


def load_checkpoint_to_sac(checkpoint: Path|CheckPoint, num_envs: int, async_vec_env: bool = True) -> tuple[SACOptimiser, dict|None]:
    """Reconstructs a fully functional SACOptimiser from a saved checkpoint file.
    
    Rebuilds the vectorized environment, restores the actor, twin Q-networks, target 
    networks, rollout buffer, alpha optimizer, and links all relevant learning rate schedulers.
    
    Args:
        path (Path): The file path of the saved checkpoint or the raw checkpoint object.
        num_envs (int): The number of parallel environments to instantiate.
        
    Returns:
        tuple[SACOptimiser, dict|None]: A tuple containing the restored SACOptimiser instance 
                                        and the user's external metadata.
                                        
    Raises:
        ValueError: If the checkpoint cannot be loaded or is not of type SACState.
        
    Example:
        >>> sac, meta = load_checkpoint_to_sac(Path("sac_model.st"), num_envs=4)
    """
    if isinstance(checkpoint, (Path,str)):
        cp_path = checkpoint
        checkpoint = CheckPoint.load(cp_path)
        if checkpoint is None:
            raise ValueError(f"Failed to load checkpoint from {cp_path}")
    if not isinstance(checkpoint, CheckPoint):
        raise ValueError(f"Checkpoint is not of type CheckPoint, got {type(checkpoint)}")
    
    state = checkpoint.state
    if not isinstance(state, SACState):
        raise ValueError(f"Checkpoint state is not of type SACState, got {type(state)}")
        
    meta_data_str = checkpoint.metadata
    meta_data: dict = loads(meta_data_str)
    
    internal_metadata: dict = meta_data.get("internal_metadata", {})
    external_metadata: dict = meta_data.get("external_metadata", {})
    
    if async_vec_env:
        env = gym.make_vec(checkpoint.env_id, num_envs=num_envs,vectorization_mode=gym.VectorizeMode.ASYNC)
    else:
        env = gym.make_vec(checkpoint.env_id, num_envs=num_envs)
    
    optimiser = SACOptimiser(env=env,
                             actor_network=state.pi,
                             q_networks=list(state.qs),
                             actor_optimizer=state.pi_optimizer,
                             q_optimizers=list(state.q_optimizers),
                             rollout_buffer=state.rollout_buffer)
    
    optimiser.q_tarrs[0].params_copy(state.q_tarrs[0].params)
    optimiser.q_tarrs[1].params_copy(state.q_tarrs[1].params)
    
    optimiser.setup_training(hyper_params=state.hyper_params,
                             obs_normalizer=state.obs_normalizer,
                             schedulers={})
                             
    if state.hyper_params.trainable_alpha and state.alpha_optimizer is not None:
        saved_alpha_state = state.alpha_optimizer.state()
        optimiser.alpha_optimizer.load_state(saved_alpha_state)
        optimiser.alpha_optimizer.initialize(state.hyper_params.log_alpha)
        
    pi_scheduler_meta = internal_metadata.get("pi_scheduler", None)
    q1_scheduler_meta = internal_metadata.get("q1_scheduler", None)
    q2_scheduler_meta = internal_metadata.get("q2_scheduler", None)
    alpha_scheduler_meta = internal_metadata.get("alpha_scheduler", None)
    
    if pi_scheduler_meta is not None:
        pi_scheduler = get_scheduler_from_meta_data(pi_scheduler_meta, optimiser.pi_optimizer)
        if pi_scheduler is not None:
            optimiser.schedulers["pi"] = pi_scheduler
            
    if q1_scheduler_meta is not None:
        q1_scheduler = get_scheduler_from_meta_data(q1_scheduler_meta, optimiser.q_optimizers[0])
        if q1_scheduler is not None:
            optimiser.schedulers["q1"] = q1_scheduler

    if q2_scheduler_meta is not None:
        q2_scheduler = get_scheduler_from_meta_data(q2_scheduler_meta, optimiser.q_optimizers[1])
        if q2_scheduler is not None:
            optimiser.schedulers["q2"] = q2_scheduler
            
    if alpha_scheduler_meta is not None and state.hyper_params.trainable_alpha:
        alpha_scheduler = get_scheduler_from_meta_data(alpha_scheduler_meta, optimiser.alpha_optimizer)
        if alpha_scheduler is not None:
            optimiser.schedulers["alpha"] = alpha_scheduler
            
    state.hyper_params.start_epoch = checkpoint.step
    return optimiser, external_metadata







import gymnasium as gym
import numpy as np
from pathlib import Path
import dataclasses
import os

# --- Imports de ton architecture ---
# Ajuste ces imports selon la structure exacte de ton projet !
from ..core import optim, nn
from ..rl import policy

def _compare(path, a, b):
    """Fonction récursive puissante pour comparer deux objets complexes."""
    if a is None and b is None:
        return
    assert type(a) is type(b), f"{path}: type mismatch ({type(a)} != {type(b)})"

    if isinstance(a, np.ndarray):
        assert a.shape == b.shape, f"{path}: shape mismatch {a.shape} != {b.shape}"
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=f"{path}: values differ")
        return

    if dataclasses.is_dataclass(a):
        _compare(path, dataclasses.asdict(a), dataclasses.asdict(b))
        return

    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys()), f"{path}: dict keys differ"
        for k in a:
            _compare(f"{path}[{k}]", a[k], b[k])
        return

    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length mismatch {len(a)} != {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            _compare(f"{path}[{i}]", ai, bi)
        return

    if isinstance(a, nn.Sequential): # Compare les paramètres des réseaux
        _compare(f"{path}.params", a.params, b.params)
        return

    if isinstance(a, optim.GradientOptimizer): # Compare l'état des optimiseurs (m, v, t...)
        _compare(f"{path}.state()", a.state(), b.state())
        return

    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        # Évite les boucles infinies sur les environnements
        if isinstance(a, (gym.Env, gym.vector.VectorEnv)):
            return 
        a_dict = {k:v for k,v in vars(a).items() if not k.startswith("_")} # Compare le public
        b_dict = {k:v for k,v in vars(b).items() if not k.startswith("_")}
        for k in a_dict:
            if k in b_dict:
                _compare(f"{path}.{k}", a_dict[k], b_dict[k])
        return

    assert a == b, f"{path}: {a!r} != {b!r}"


def test_openaies():
    print("🧪 Test OpenAIES...")
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = nn.Sequential([nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, act_dim)])
    opt = optim.Adam(alpha=1e-3)
    es = OpenAIES(env=env, network=net, optimizer=opt)
    es.setup_training(hyper_params=OpenAIES.HyperParams(), obs_normalizer=None, schedulers={})

    path = Path("test_es.st")
    save_checkpoint_from_openaies(es, path, step=15, meta_data={"test": "es_ok"})
    loaded_es, meta = load_checkpoint_to_openaies(path)

    assert meta["test"] == "es_ok"
    _compare("es_network", es.main_network, loaded_es.main_network)
    _compare("es_optimizer", es.optimizer, loaded_es.optimizer)
    print("✅ OpenAIES OK\n")
    if path.exists(): os.remove(path)


def test_ppo():
    print("🧪 Test PPOptimiser...")
    num_envs = 2
    env = gym.make_vec("Pendulum-v1", num_envs=num_envs)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    actor, critic = policy.make_actor_critic_tuple(obs_dim, act_dim, hidden_layers_count=1, hidden_dims=16)
    a_opt, c_opt = optim.AdamW(alpha=3e-4), optim.AdamW(alpha=3e-4)

    ppo = PPOptimiser(env=env, networks=(actor, critic), actor_optimizer=a_opt, critic_optimizer=c_opt)
    hp = PPOptimiser.HyperParams(rollout_length=128, mini_batch_size=32)
    ppo.setup_training(hyper_params=hp, obs_normalizer=None, schedulers={})

    path = Path("test_ppo.st")
    save_checkpoint_from_ppo(ppo, path, step=42, meta_data={"test": "ppo_ok"})
    loaded_ppo, meta = load_checkpoint_to_ppo(path, num_envs=num_envs)

    assert meta["test"] == "ppo_ok"
    assert loaded_ppo.hyper_params.start_epoch == 42
    _compare("ppo_actor", ppo.actor, loaded_ppo.actor)
    _compare("ppo_critic", ppo.critic, loaded_ppo.critic)
    _compare("ppo_a_opt", ppo.actor_optimizer, loaded_ppo.actor_optimizer)
    _compare("ppo_c_opt", ppo.critic_optimizer, loaded_ppo.critic_optimizer)
    print("✅ PPOptimiser OK\n")
    if path.exists(): os.remove(path)


def test_sac():
    print("🧪 Test SACOptimiser...")
    num_envs = 2
    env = gym.make_vec("Pendulum-v1", num_envs=num_envs)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    actor, q1, q2 = policy.make_q_and_actor_tuple(obs_dim, act_dim, hidden_layers_count=1, hidden_dims=16)
    pi_opt = optim.Adam(alpha=3e-4)
    q1_opt, q2_opt = optim.Adam(alpha=3e-4), optim.Adam(alpha=3e-4)

    sac = SACOptimiser(env=env, actor_network=actor, q_networks=[q1, q2], 
                       actor_optimizer=pi_opt, q_optimizers=[q1_opt, q2_opt])
    
    hp = SACOptimiser.HyperParams(rollout_size=1000, step_amount=16)
    sac.setup_training(hyper_params=hp, obs_normalizer=None, schedulers={})

    # On modifie un peu les réseaux cibles manuellement pour tester si la copie marche bien
    sac.q_tarrs[0].init_random_params()
    
    path = Path("test_sac.st")
    save_checkpoint_from_sac(sac, path, step=99, meta_data={"version": 1.0})
    loaded_sac, meta = load_checkpoint_to_sac(path, num_envs=num_envs)

    assert meta["version"] == 1.0
    assert loaded_sac.hyper_params.start_epoch == 99
    
    # 1. Test des réseaux
    _compare("sac_pi", sac.pi, loaded_sac.pi)
    _compare("sac_q1", sac.qs[0], loaded_sac.qs[0])
    _compare("sac_q2", sac.qs[1], loaded_sac.qs[1])
    _compare("sac_qtarr1", sac.q_tarrs[0], loaded_sac.q_tarrs[0]) # Test crucial pour SAC
    
    # 2. Test des optimiseurs
    _compare("sac_pi_opt", sac.pi_optimizer, loaded_sac.pi_optimizer)
    _compare("sac_q1_opt", sac.q_optimizers[0], loaded_sac.q_optimizers[0])
    _compare("sac_alpha_opt", sac.alpha_optimizer, loaded_sac.alpha_optimizer)

    # 3. Test du buffer
    _compare("sac_buffer_obs", sac.rollout_buffer.obs, loaded_sac.rollout_buffer.obs)
    
    print("✅ SACOptimiser OK\n")
    if path.exists(): os.remove(path)


if __name__ == "__main__":
    print("🚀 Lancement des tests de sérialisation...\n")
    test_openaies()
    test_ppo()
    test_sac()
    print("🎉 Tous les tests sont passés avec succès !")