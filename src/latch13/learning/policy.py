"""Trained policies of ``scripts/train_rl.py``: actor settings, checkpoints and the theta they choose.

``actor_config`` is the single definition of the actor used in training, so a
checkpoint can be rebuilt exactly. ``theta_function`` gives theta(observation)
for any policy; ``constant_theta`` the fixed theta of a ``--constant-theta``
policy, which the plain closed-loop runner can use.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from leap_c.parameters.utils import broadcast_default_param
from mpc_sac.rl.mpc_actor import HierachicalMPCActor, HierachicalMPCActorConfig
from mpc_sac.utils import torch_gym
from torch import nn

from latch13.controllers.leapc_ocp import DiagonalTerminalTheta


def actor_config(constant_theta=False):
    """The actor of SAC-ZOP training (mpc-sac's example settings)."""
    cfg = HierachicalMPCActorConfig()
    cfg.noise = "param"
    cfg.extractor_name = "scaling"
    cfg.distribution_name = "squashed_gaussian"
    cfg.residual = True  # start at the initial theta, not the middle of its bounds
    # No hidden layers: the actor is a single learned theta, whatever the observation.
    cfg.mlp.hidden_dims = None if constant_theta else (256, 256, 256)
    return cfg


class ThetaOnly(nn.Module):
    """Stands in for the MPC when only theta is needed (no solver is built)."""

    def __init__(self, theta=None):
        super().__init__()
        theta = theta if theta is not None else DiagonalTerminalTheta()
        self._space, self._default = theta.space(), theta.defaults()

    @property
    def param_space(self):
        return self._space

    def default_param(self, obs=None):
        return broadcast_default_param(self._default, obs)


def run_config(run):
    return json.loads((Path(run) / "run_config.json").read_text(encoding="utf-8"))


def best_checkpoint(run):
    """(step, validation score) of the saved checkpoint with the best validation score."""
    run = Path(run)
    log = pd.read_csv(run / "val_log.csv").drop_duplicates("timestamp", keep="last")
    saved = {int(path.name.split("_")[0]) for path in (run / "ckpts").glob("*_pi.ckpt")}
    log = log[log.timestamp.isin(saved)]
    if log.empty:
        raise FileNotFoundError(f"No validated checkpoint in {run}")
    best = log.loc[log.score.idxmax()]
    return int(best.timestamp), float(best.score)


def load_actor(run, step, observation_space, action_space, controller=None):
    """The trained actor of ``run`` at checkpoint ``step``, in eval mode.

    ``controller`` is the MPC the actor drives (``ControllerFromPlanner``), or
    None when only theta is needed.
    """
    run = Path(run)
    args = run_config(run)["args"]
    dtype = getattr(torch, args.get("dtype", "float64"))
    actor = HierachicalMPCActor(actor_config(args.get("constant_theta", False)), observation_space,
                                action_space, controller if controller is not None else ThetaOnly())
    actor = actor.to(dtype=dtype)
    actor.load_state_dict(torch.load(run / "ckpts" / f"{step}_pi.ckpt", weights_only=False))
    return actor.eval()


def theta_function(actor):
    """theta(observation) of an actor, deterministic, as float64 tensors with a batch of 1."""
    dtype = next(iter(actor.state_dict().values())).dtype

    def theta(observation):
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observation)[None], dtype=dtype)
            param = actor(obs, None, deterministic=True, only_param=True).param
        return {name: value.to(torch.float64)
                for name, value in torch_gym.unflatten(actor.param_space, param).items()}
    return theta


def constant_theta(run, step, observation_space, action_space):
    """The fixed theta of a ``--constant-theta`` policy, as numpy arrays."""
    if not run_config(run)["args"].get("constant_theta", False):
        raise ValueError(f"{run} learned a road-dependent theta, which changes every step; "
                         "use scripts/compare.py to drive it")
    actor = load_actor(run, step, observation_space, action_space)
    theta = theta_function(actor)(np.zeros(observation_space.shape))
    return {name: value[0].numpy() for name, value in theta.items()}
