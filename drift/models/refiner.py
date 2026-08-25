"""Refiner: reconcile static/dynamic seams via spatiotemporal interaction.

REUSED from OccProphet (Sec. 3.4). Origin: OccProphet (ICLR 2025).

Concatenates the observation and forecasted-future latents along time, runs them jointly through
one `EfficientAggregation4D`, and keeps only the future slice -- letting attention/conv in E4A
smooth over the seam between real observed history and the (independently forecasted) static and
instance-splatted dynamic future.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from drift.models.e4a import EfficientAggregation4D


class Refiner(nn.Module):
    """Reconcile static/dynamic seams via spatiotemporal interaction. REUSED (OccProphet Sec. 3.4).

    Args:
        channels: Channel width of both `obs_latent` and `future_latent` (and of the output).
        in_timesteps: `T_p`.
        out_timesteps: `T_o`.
        **e4a_kwargs: Forwarded to the internal `EfficientAggregation4D`
            (`embed_dims`/`out_channels` default to `channels` unless overridden here).
    """

    def __init__(
        self, channels: int, in_timesteps: int, out_timesteps: int, **e4a_kwargs: Any
    ) -> None:
        super().__init__()
        self.channels = channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps

        kwargs = dict(e4a_kwargs)
        kwargs.setdefault("embed_dims", channels)
        kwargs.setdefault("out_channels", channels)
        kwargs.setdefault("multiscale_output", False)
        if kwargs["multiscale_output"]:
            raise ValueError(
                "Refiner requires a single-tensor E4A output; pass multiscale_output=False "
                "(or omit it) in e4a_kwargs."
            )

        self.e4a = EfficientAggregation4D(
            in_channels=channels, timesteps=in_timesteps + out_timesteps, **kwargs
        )

    def forward(self, obs_latent: Tensor, future_latent: Tensor) -> Tensor:
        """
        Args:
            obs_latent: (B, T_p, C, X, Y, Z), C == channels, T_p == in_timesteps.
            future_latent: (B, T_o, C, X, Y, Z), T_o == out_timesteps.

        Returns:
            (B, T_o, C, X, Y, Z) -- the refined future slice only.
        """
        if obs_latent.dim() != 6 or future_latent.dim() != 6:
            raise ValueError(
                "Refiner expects both obs_latent and future_latent as 6D (B,T,C,X,Y,Z) "
                f"tensors, got {tuple(obs_latent.shape)} and {tuple(future_latent.shape)}."
            )
        if obs_latent.shape[1] != self.in_timesteps:
            raise ValueError(
                f"Refiner: obs_latent has T={obs_latent.shape[1]} but in_timesteps="
                f"{self.in_timesteps}."
            )
        if future_latent.shape[1] != self.out_timesteps:
            raise ValueError(
                f"Refiner: future_latent has T={future_latent.shape[1]} but out_timesteps="
                f"{self.out_timesteps}."
            )
        if obs_latent.shape[2] != self.channels or future_latent.shape[2] != self.channels:
            raise ValueError(
                f"Refiner: expected channels={self.channels}, got obs C="
                f"{obs_latent.shape[2]}, future C={future_latent.shape[2]}."
            )

        x = torch.cat([obs_latent, future_latent], dim=1)  # (B, T_p+T_o, C, X, Y, Z)
        out = self.e4a(x)
        return out[:, self.in_timesteps :]
