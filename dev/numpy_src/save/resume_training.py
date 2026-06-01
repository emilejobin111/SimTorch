"""Utility entry point for resuming PPO training from checkpoints.

This module recreates the vectorized environment, loads the saved PPO state, and
continues training from the recovered epoch.
"""

import gymnasium as gym
from ..rl.ppoptimiser import PPOptimiser
from .save_nn import get_checkpoint_dir

TRAINING_NAME = None

def main():
    """Run the demonstration entry point defined by this module.
    
    Returns:
        None: This method performs its work in place and returns no value.
    
    Raises:
        e: Raised if the operation cannot be completed successfully.
    
    Example:
        >>> result = main()
    """
    print("🚀 Initialisation de l'environnement vectorisé...")
    env = gym.make_vec("Ant-v5", num_envs=16, vectorization_mode="async")
    checkpoint_dir = get_checkpoint_dir(TRAINING_NAME)
    
    print("Demande au PPOptimiser de charger ses checkpoints...")
    
    ppo, hp, obs_norm = PPOptimiser.load_checkpoint(env, checkpoint_dir)
    
    print(f"Reprise de l'entraînement à partir de l'epoch {hp.start_epoch}...")
    try:
        ppo.train(hyper_params=hp, obs_normalizer=obs_norm)
    except Exception as e:
        raise e
    finally:
        env.close()

if __name__ == "__main__":
    main()
