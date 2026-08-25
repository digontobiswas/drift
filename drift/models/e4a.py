"""Efficient 4D Aggregation (E4A).

REUSED from OccProphet (Sec. 3.2.1). Origin: OccProphet (ICLR 2025).

A symmetric 3D UNet applied per-timestep (T folded into the batch dimension for all
convolutions) with :class:`~drift.models.taf.TriplingAttentionFusion` plugins inserted on the
upsampling path at every scale, and additive skip connections from the encoder. Used twice in
DRIFT: as the :class:`~drift.models.observer.Observer` body and as the
:class:`~drift.models.refiner.Refiner` body.

Documented simplification: rather than fixed-stride `Conv3d`/`ConvTranspose3d` down/up-sampling
(which breaks for latent grids whose spatial size is not a power of two along every axis -- the
default latent Z=10 halves to 5, then 2, then 1 -- and which would require the caller to pass a
static `latent_size` that E4A's constructor does not accept), each down/up step resizes via
`F.interpolate` (nearest for downsampling, trilinear for upsampling) to a size computed
dynamically from the actual input shape, followed by a learned `Conv3d` to change channel width.
This keeps E4A correct for arbitrary (including odd or tiny) spatial sizes while remaining a
"symmetric UNet with doubling channels per level and skip connections" as specified.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: `import torch` alone need not bind this submodule
from torch import Tensor, nn

from drift.models.taf import TriplingAttentionFusion


def _auto_num_heads(channels: int, max_heads: int = 8) -> int:
    for h in (max_heads, 4, 2, 1):
        if h <= channels and channels % h == 0:
            return h
    return 1


def _half_size(size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(max(1, d // 2) for d in size)  # type: ignore[return-value]


class _ConvBnAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(_auto_num_heads(out_ch, 8), out_ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class EfficientAggregation4D(nn.Module):
    """Efficient 4D Aggregation (UNet over the latent volume). REUSED from OccProphet (Sec. 3.2.1).

    Args:
        in_channels: Channels of the input volume.
        embed_dims: Channel width immediately after the input projection (level 0 of the UNet);
            doubles at every downsample level.
        out_channels: Output channel count. Defaults to `embed_dims` when None.
        downsample_layers: Number of downsample/upsample levels.
        in_proj_kernel_size: Kernel size of the input projection conv.
        upsample_plugins: Optional explicit list of `downsample_layers` TAF-like modules
            (callable as `module(x)` with `x: (B,T,C,X,Y,Z)`, one per upsample level, ordered
            from the bottleneck outward). If None, default `TriplingAttentionFusion` modules are
            constructed automatically with `embed_dims` matching each level's channel count.
        timesteps: Documented expected number of frames; the actual `T` is read from the input
            at forward time, so this is informational only.
        multiscale_output: If True, `forward` returns a list of feature maps (one per upsample
            level, coarsest-to-finest, ending with the final output). If False (default), only
            the final `(B,T,out_channels,X,Y,Z)` tensor is returned.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        out_channels: Optional[int] = None,
        downsample_layers: int = 3,
        in_proj_kernel_size: int = 1,
        upsample_plugins: Optional[List[nn.Module]] = None,
        timesteps: int = 3,
        multiscale_output: bool = False,
    ) -> None:
        super().__init__()
        if downsample_layers < 1:
            raise ValueError(
                f"EfficientAggregation4D: downsample_layers must be >= 1, got "
                f"{downsample_layers}."
            )
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.out_channels = out_channels if out_channels is not None else embed_dims
        self.downsample_layers = downsample_layers
        self.timesteps = timesteps
        self.multiscale_output = multiscale_output

        self.in_proj = nn.Conv3d(
            in_channels, embed_dims, in_proj_kernel_size, padding=in_proj_kernel_size // 2
        )

        # channel count at each encoder level: ch[0]=embed_dims .. ch[L]=embed_dims*2^L
        self.level_channels = [embed_dims * (2 ** k) for k in range(downsample_layers + 1)]

        self.downs = nn.ModuleList(
            [
                _ConvBnAct(self.level_channels[k - 1], self.level_channels[k])
                for k in range(1, downsample_layers + 1)
            ]
        )

        self.up_reduce = nn.ModuleList()
        self.up_fuse = nn.ModuleList()
        for k in range(downsample_layers, 0, -1):
            c_in = self.level_channels[k]
            c_out = self.level_channels[k - 1]
            self.up_reduce.append(nn.Conv3d(c_in, c_out, kernel_size=1))
            self.up_fuse.append(_ConvBnAct(2 * c_out, c_out))

        if upsample_plugins is not None:
            if len(upsample_plugins) != downsample_layers:
                raise ValueError(
                    f"EfficientAggregation4D: upsample_plugins must have exactly "
                    f"downsample_layers={downsample_layers} entries, got "
                    f"{len(upsample_plugins)}."
                )
            self.plugins = nn.ModuleList(upsample_plugins)
        else:
            self.plugins = nn.ModuleList(
                [
                    TriplingAttentionFusion(
                        embed_dims=self.level_channels[k - 1],
                        num_heads=_auto_num_heads(self.level_channels[k - 1]),
                        timesteps=timesteps,
                        residual=True,
                    )
                    for k in range(downsample_layers, 0, -1)
                ]
            )

        self.out_proj = nn.Conv3d(embed_dims, self.out_channels, kernel_size=1)

        # Set by DRIFT.__init__ (propagated to every submodule declaring this attribute).
        # See DRIFT._ckpt. Checkpointing matters *per stage* here rather than for the
        # module as a whole: an outer checkpoint around the whole refiner still has to
        # materialise every E4A activation at once when backward recomputes it, which is
        # exactly where a 16 GB V100 runs out. Checkpointing each stage separately keeps
        # the recompute peak to one stage even when an outer checkpoint re-runs this
        # forward.
        self.grad_checkpoint = False

    def _ckpt(self, fn: Any, *args: Any) -> Any:
        """Gradient-checkpoint ``fn(*args)`` when enabled. Mirrors ``DRIFT._ckpt``."""
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return fn(*args)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

    def _down_stage(self, k: int, x: Tensor, size: Tuple[int, int, int]) -> Tensor:
        x = F.interpolate(x, size=size, mode="nearest")
        return self.downs[k](x)

    def _up_stage(
        self, i: int, x: Tensor, skip: Tensor, target_size: Tuple[int, int, int], b: int, t: int
    ) -> Tensor:
        """One decoder level: upsample, reduce, fuse the skip, then the TAF plugin.

        Grouped into a single function so the whole level -- including the widest
        tensor in it, the post-``cat`` fusion input -- can be checkpointed as a unit.
        """
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        x = self.up_reduce[i](x)
        x = torch.cat([x, skip], dim=1)
        x = self.up_fuse[i](x)

        c_cur = x.shape[1]
        x_5d = x.reshape(b, t, c_cur, *target_size)
        x_5d = self.plugins[i](x_5d)
        return x_5d.reshape(b * t, c_cur, *target_size)

    def forward(self, vox_feats: Tensor) -> Union[Tensor, List[Tensor]]:
        """
        Args:
            vox_feats: (B, T, C, X, Y, Z) with C == in_channels.

        Returns:
            (B, T, out_channels, X, Y, Z), or a list of such tensors (varying channel counts)
            if `multiscale_output=True`.
        """
        if vox_feats.dim() != 6:
            raise ValueError(
                f"EfficientAggregation4D expects a 6D (B,T,C,X,Y,Z) tensor, got shape "
                f"{tuple(vox_feats.shape)}."
            )
        b, t, c, x_dim, y_dim, z_dim = vox_feats.shape
        if c != self.in_channels:
            raise ValueError(
                f"EfficientAggregation4D: input has {c} channels but in_channels="
                f"{self.in_channels}."
            )

        # precompute the spatial size at every level
        sizes: List[Tuple[int, int, int]] = [(x_dim, y_dim, z_dim)]
        for _ in range(self.downsample_layers):
            sizes.append(_half_size(sizes[-1]))

        x = vox_feats.reshape(b * t, c, x_dim, y_dim, z_dim)
        x = self.in_proj(x)

        skips = [x]
        for k in range(1, self.downsample_layers + 1):
            x = self._ckpt(self._down_stage, k - 1, x, sizes[k])
            skips.append(x)

        outputs: List[Tensor] = []
        for i in range(self.downsample_layers):
            level = self.downsample_layers - i  # current source level (counting down)
            target_size = sizes[level - 1]
            x = self._ckpt(self._up_stage, i, x, skips[level - 1], target_size, b, t)
            outputs.append(x)

        final = self.out_proj(x)
        final = final.reshape(b, t, self.out_channels, x_dim, y_dim, z_dim)

        if self.multiscale_output:
            multi = [
                o.reshape(b, t, o.shape[1], *o.shape[-3:]) for o in outputs[:-1]
            ] + [final]
            return multi
        return final
