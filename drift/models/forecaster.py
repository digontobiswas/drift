"""DecoupledForecaster: static ego-warp + instance-query dynamic path, gate-merged.

★ NOVEL assembly. Contrast with DFIT-OccWorld, which decouples static/dynamic forecasting using
a DENSE per-voxel flow field; here the dynamic branch is entirely instance-query based (see
`drift/models/instance_path.py`).

Wires together `StaticForecastPath`, `InstanceQueryExtractor` + `MotionForecaster` +
`InstanceSplatter`, and an optional `ConditionalForecaster` that supplies scene-conditioned
context to both branches, then merges the two branches with a learned per-voxel gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.utils.checkpoint  # explicit: `import torch` alone need not bind this submodule
from torch import Tensor, nn

from drift.models.condition import ConditionalForecaster
from drift.models.instance_path import (
    InstanceQueryExtractor,
    InstanceSplatter,
    InstanceState,
    MotionForecaster,
)
from drift.models.static_path import StaticForecastPath


class DecoupledForecaster(nn.Module):
    """★ NOVEL. Static scene by ego-warp + dynamic agents in instance-query space, gate-merged.

    Args:
        channels: Latent channel width, shared by all sub-modules' inputs/outputs.
        num_future: `T_o`, number of future frames to forecast (== `MotionForecaster.num_future`
            and the length of `future_ego`'s time axis).
        static_cfg: Kwargs forwarded to `StaticForecastPath` (`channels` is auto-injected from
            this module's `channels` and must not conflict if also present).
        instance_cfg: Optional nested config with up to three sub-dicts -- `"extractor"`
            (-> `InstanceQueryExtractor`, `in_channels` auto-injected), `"motion"`
            (-> `MotionForecaster`, `embed_dims`/`num_future` auto-injected), and `"splatter"`
            (-> `InstanceSplatter`, `embed_dims`/`out_channels` auto-injected). Missing sub-dicts
            use each class's defaults.
        condition_cfg: If given, kwargs forwarded to `ConditionalForecaster`. Since
            `ConditionalForecaster` is built eagerly at `__init__` time (not lazily on first
            forward), `condition_cfg` **must** include `"in_timesteps"` (== `T_p`);
            `in_channels`/`out_timesteps` default to `channels`/`num_future` when omitted.
        merge: `"gate"` (default) computes a learned per-voxel gate `a = sigmoid(gate(concat(
            static, dyn, dyn_occ)))` and returns `static*(1-a) + dyn*a`. `"sum"` simply adds the
            two branches (kept for ablations).
    """

    def __init__(
        self,
        channels: int,
        num_future: int,
        static_cfg: Optional[Dict[str, Any]] = None,
        instance_cfg: Optional[Dict[str, Any]] = None,
        condition_cfg: Optional[Dict[str, Any]] = None,
        merge: Literal["gate", "sum"] = "gate",
        latent_size: Optional[Tuple[int, int, int]] = None,
        point_cloud_range: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if merge not in ("gate", "sum"):
            raise ValueError(f"DecoupledForecaster: merge must be 'gate' or 'sum', got {merge!r}.")
        self.channels = channels
        self.num_future = num_future
        self.merge = merge

        # The static path and the instance splatter must rasterize onto the SAME voxel grid.
        # Injecting the grid here -- rather than relying on each sub-config to repeat it -- makes
        # a silent mismatch structurally impossible. Previously the splatter could fall back to
        # its (128, 128, 10) default while the static path used the configured grid, which only
        # surfaced as a shape error at the merge.
        def _inject_grid(kwargs: Dict[str, Any], name: str) -> Dict[str, Any]:
            for key, val in (
                ("latent_size", latent_size),
                ("point_cloud_range", point_cloud_range),
            ):
                if val is None:
                    continue
                existing = kwargs.get(key)
                if existing is not None and tuple(existing) != tuple(val):
                    raise ValueError(
                        f"DecoupledForecaster: {name}[{key!r}]={existing} conflicts with the "
                        f"forecaster-level {key}={val}. Omit it from the sub-config."
                    )
                kwargs[key] = val
            return kwargs

        self._inject_grid = _inject_grid
        static_cfg = _inject_grid(dict(static_cfg or {}), "static_cfg")
        if "channels" in static_cfg and static_cfg["channels"] not in (None, channels):
            raise ValueError(
                f"DecoupledForecaster: static_cfg['channels']={static_cfg['channels']} "
                f"conflicts with channels={channels}."
            )
        static_cfg["channels"] = channels
        self.static_path = StaticForecastPath(**static_cfg)

        instance_cfg = dict(instance_cfg or {})
        extractor_kwargs = dict(instance_cfg.get("extractor", {}))
        if "in_channels" in extractor_kwargs and extractor_kwargs["in_channels"] != channels:
            raise ValueError(
                "DecoupledForecaster: instance_cfg['extractor']['in_channels'] must equal "
                f"channels={channels} or be omitted."
            )
        extractor_kwargs["in_channels"] = channels
        self.instance_extractor = InstanceQueryExtractor(**extractor_kwargs)
        embed_dims = self.instance_extractor.embed_dims

        condition_cfg_kwargs: Optional[Dict[str, Any]] = None
        if condition_cfg is not None:
            condition_cfg_kwargs = dict(condition_cfg)
            condition_cfg_kwargs.setdefault("in_channels", channels)
            condition_cfg_kwargs.setdefault("out_timesteps", num_future)
            if "in_timesteps" not in condition_cfg_kwargs:
                raise ValueError(
                    "DecoupledForecaster: condition_cfg must specify 'in_timesteps' (== T_p); "
                    "ConditionalForecaster is built eagerly at __init__ time and cannot infer "
                    "it from the forward-time input shape."
                )
            self.condition: Optional[ConditionalForecaster] = ConditionalForecaster(
                **condition_cfg_kwargs
            )
        else:
            self.condition = None

        motion_kwargs = dict(instance_cfg.get("motion", {}))
        for key, val in (("embed_dims", embed_dims), ("num_future", num_future)):
            if key in motion_kwargs and motion_kwargs[key] != val:
                raise ValueError(
                    f"DecoupledForecaster: instance_cfg['motion'][{key!r}]={motion_kwargs[key]} "
                    f"conflicts with the required value {val}."
                )
            motion_kwargs[key] = val
        if self.condition is not None:
            motion_kwargs.setdefault("scene_condition_dims", channels)
        self.motion_forecaster = MotionForecaster(**motion_kwargs)

        splatter_kwargs = _inject_grid(
            dict(instance_cfg.get("splatter", {})), "instance_cfg['splatter']"
        )
        for key, val in (("embed_dims", embed_dims), ("out_channels", channels)):
            if key in splatter_kwargs and splatter_kwargs[key] != val:
                raise ValueError(
                    f"DecoupledForecaster: instance_cfg['splatter'][{key!r}]="
                    f"{splatter_kwargs[key]} conflicts with the required value {val}."
                )
            splatter_kwargs[key] = val
        self.instance_splatter = InstanceSplatter(**splatter_kwargs)

        if merge == "gate":
            self.gate_conv = nn.Conv3d(2 * channels + 1, 1, kernel_size=1)
        else:
            self.gate_conv = None

        # Set by DRIFT.__init__ from cfg.grad_checkpoint. Deliberately not a constructor
        # argument, so existing callers/tests that build a DecoupledForecaster directly
        # keep working unchanged. See DRIFT._ckpt for the full rationale.
        self.grad_checkpoint = False

    def _ckpt(self, fn: Any, *args: Any) -> Any:
        """Gradient-checkpoint ``fn(*args)`` when enabled. Mirrors ``DRIFT._ckpt``."""
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return fn(*args)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

    def _gate_merge(self, static_latent: Tensor, dyn_latent: Tensor, dyn_occ: Tensor) -> Tensor:
        """Gated blend of the static and dynamic futures.

        Split out of ``forward`` so it can be gradient-checkpointed as a unit: the
        ``cat`` below is the widest tensor in the model (``2*C+1`` channels over the
        full 5D latent grid, ~1 GB at C=128/T_o=6) and exists only to be consumed by
        a 1x1 conv, so recomputing it in backward is cheap next to keeping it resident.
        """
        b, t_o, _, x_dim, y_dim, z_dim = static_latent.shape
        gate_in = torch.cat([static_latent, dyn_latent, dyn_occ], dim=2)
        gate_in = gate_in.reshape(b * t_o, 2 * self.channels + 1, x_dim, y_dim, z_dim)
        a = torch.sigmoid(self.gate_conv(gate_in))
        a = a.reshape(b, t_o, 1, x_dim, y_dim, z_dim)
        return static_latent * (1 - a) + dyn_latent * a

    def forward(self, obs_latent: Tensor, future_ego: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Args:
            obs_latent: (B, T_p, C, X, Y, Z), C == channels.
            future_ego: (B, T_o, 4, 4), T_o == num_future.

        Returns:
            future_latent: (B, T_o, C, X, Y, Z)
            aux: dict with keys 'instance_states' (List[InstanceState]), 'dyn_occ',
                'static_latent', 'dyn_latent', 'present_state' (InstanceState).
        """
        if obs_latent.dim() != 6:
            raise ValueError(
                f"DecoupledForecaster expects obs_latent (B,T_p,C,X,Y,Z), got "
                f"{tuple(obs_latent.shape)}."
            )
        b, t_p, c, x_dim, y_dim, z_dim = obs_latent.shape
        if c != self.channels:
            raise ValueError(
                f"DecoupledForecaster: obs_latent has {c} channels but channels={self.channels}."
            )
        if future_ego.shape[1] != self.num_future:
            raise ValueError(
                f"DecoupledForecaster: future_ego has T_o={future_ego.shape[1]} but "
                f"num_future={self.num_future}."
            )

        present = obs_latent[:, -1]  # (B,C,X,Y,Z)
        static_latent = self._ckpt(self.static_path, present, future_ego)  # (B,T_o,C,X,Y,Z)

        scene_cond_vec: Optional[Tensor] = None
        if self.condition is not None:
            cond_latent = self._ckpt(self.condition, obs_latent)  # (B,T_o,C,X,Y,Z)
            static_latent = static_latent + cond_latent
            scene_cond_vec = cond_latent.mean(dim=(1, 3, 4, 5))  # (B,C)

        present_state = self.instance_extractor(present)
        if scene_cond_vec is not None:
            future_states: List[InstanceState] = self.motion_forecaster(
                present_state, scene_cond=scene_cond_vec
            )
        else:
            future_states = self.motion_forecaster(present_state)

        dyn_latent, dyn_occ = self.instance_splatter(future_states)  # (B,T_o,C,X,Y,Z), (B,T_o,1,X,Y,Z)

        if self.merge == "gate":
            out = self._ckpt(self._gate_merge, static_latent, dyn_latent, dyn_occ)
        else:
            out = static_latent + dyn_latent

        aux: Dict[str, Any] = {
            "instance_states": future_states,
            "dyn_occ": dyn_occ,
            "static_latent": static_latent,
            "dyn_latent": dyn_latent,
            "present_state": present_state,
        }
        return out, aux
