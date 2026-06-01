"""Minimal runnable training example with a named checkpoint folder.

Run from repo root:
`MPLCONFIGDIR=/tmp .venv/bin/python dev/tool/run_as_package.py dev/src/SimTorch/rl_worker/test_run.py`
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import gymnasium as gym

from SimTorch.core import nn, optim
from SimTorch.rl.openaies import OpenAIES
from SimTorch.rl.utils import NoNormalizer
from SimTorch.rl_worker.rl_worker import OpenAIESWorker
from SimTorch.save.save_nn import get_checkpoint_dir


name = "charlicharlikriky"


def make_network(env: gym.Env) -> nn.Sequential:
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network = nn.Sequential([
        nn.Linear(obs_dim, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, action_dim),
    ])
    network.init_random_params()
    return network


def run() -> None:
    env = gym.make("CartPole-v1")
    try:
        network = make_network(env)
        worker = OpenAIESWorker()
        hyper_params = OpenAIES.HyperParams(num_epoch=1, num_episodes=1)

        worker.setup(
            env=env,
            hyper_params=hyper_params,
            network=network,
            optimizer=optim.Adam(alpha=0.01, maximise=True),
            obs_normalizer=NoNormalizer(),
            training_name=name,
        )
        worker.train()
        print(f"Checkpoint folder: {get_checkpoint_dir(name)}")
    finally:
        env.close()


if __name__ == "__main__":
    run()
