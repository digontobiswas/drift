# DRIFT — Design & Interface Specification

**DRIFT**: *Decoupled Robust Instance-Flow Temporal occupancy world model.*

LiDAR–Camera fusion world model for 4D semantic occupancy forecasting.

This document is the **single source of truth** for module interfaces. Every implementation
file must conform to the shapes, names, and contracts below. Do not deviate silently; if a
contract seems wrong, implement it as specified and add a `# SPEC-DEVIATION:` comment.

---

## 0. Provenance & novelty map

Three reference works. We reimplement clean-room (no copied code), and cite in docstrings.

| Component | Origin | Status |
|---|---|---|
| E4A (4D UNet aggregation) | OccProphet (ICLR 2025) | **REUSED** — baseline component |
| TAF (Tripling-Attention Fusion) | OccProphet | **REUSED** |
| Conditional Forecaster (hypernetwork) | OccProphet | **REUSED**, repurposed |
| Refiner (E4A on [past, future]) | OccProphet | **REUSED** |
| Coarse Voxel Query Generator | Doracamom (TCSVT 2026) | **MODIFIED** — real 3D LiDAR volume, not height-broadcast |
| Cross-Modal BEV-Voxel Fusion | Doracamom | **MODIFIED** — two-sided gate (LiDAR is not a weak prior) |
| Decoupled static/dynamic forecasting | DFIT-OccWorld (2412.13772) | **DIFFERENT** — instance-query, not dense voxel flow |
| **Instance-query dynamic path** | — | **★ NOVEL** |
| **CMLI (Cross-Modal Latent Imagination)** | — | **★ NOVEL** |
| **Per-voxel uncertainty head** | — | **★ NOVEL** |

### Why the LiDAR substitution is not a trivial sensor swap
Doracamom broadcasts the radar BEV vector to every height (`unsqueeze(-1).repeat(..., Z)`) in
**two** places (query generator + fusion), because 4D radar has almost no elevation resolution.
Doing this with LiDAR discards its single strongest signal. DRIFT replaces both broadcasts with
a genuine 3D volume from a sparse/pillar 3D encoder. This is a documented design change, not a
port.

### Why instance-query dynamics fit the GT better than dense flow
Cam4DOcc's flow GT is **not** a per-voxel displacement field. It is a *backward centroid-offset*
field: every voxel of instance *k* at frame *t* stores `centroid_k(t-1) - own_index(t)`, in units
of 0.8 m voxels, with 255 = ignore. The GT is therefore already instance-centric — a set of
per-instance translations rendered densely. Predicting it with a dense flow field spends capacity
re-deriving a structure that is known a priori. An instance-query formulation predicts the
per-instance translation directly and renders it, which is both sparser (faster) and better matched
to the supervision.

---

## 1. Global constants and conventions

```python
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]   # x_min,y_min,z_min,x_max,y_max,z_max
OCC_SIZE          = [512, 512, 40]      # full-resolution GT grid, 0.2 m voxels
LATENT_DOWNSAMPLE = 4                   # -> latent grid
LATENT_SIZE       = [128, 128, 10]      # 0.8 m voxels; ALL internal features live here
EMPTY_IDX         = 0
```

**Time convention.** `T_p` past frames (including present), `T_f` future frames to evaluate,
`T_o` output frames the model emits (`T_o >= T_f + 1`, extra slack frames aid instance association).

| Preset | `T_p` | `T_f` | `T_o` | Horizon | Protocol |
|---|---|---|---|---|---|
| `cam4docc_2s` (default) | 3 | 4 | 6 | +2.0 s | Cam4DOcc / OccProphet comparable |
| `extended_3s` | 3 | 6 | 8 | +3.0 s | not directly comparable to Cam4DOcc numbers |

nuScenes keyframes are 2 Hz → **0.5 s per step**. Never claim 3 s under the `cam4docc_2s` preset.

**Present frame index** is `T_p - 1` in the input sequence. All past features are warped into the
**present** ego/LiDAR frame; the present frame is never warped.

**Tensor layout.** Voxel features are always `(B, T, C, X, Y, Z)` — batch, time, channel, then
spatial in x,y,z. BEV features are `(B, T, C, X, Y)`. Flatten order for queries is `(x, y, z)`
with z fastest: `idx = (x * Y + y) * Z + z`.

**Ego motion.** `ego_motion[:, t]` is the 4×4 transform `T_{frame t+1 <- frame t}` acting on
points expressed in frame `t`'s LiDAR coordinates. Entry `T_p-1` and beyond may be identity.

---

## 2. Module contracts

Every module is one file. Every public class takes a config dataclass or explicit kwargs, has
full type hints, and a docstring naming its origin (REUSED / MODIFIED / NOVEL) per §0.

### 2.1 `drift/models/encoders/camera_encoder.py`

```python
class CameraEncoder(nn.Module):
    """Multi-view images -> latent voxel volume via LSS depth-splat lift. REUSED (OccProphet/BEVDet)."""
    def __init__(self, backbone: str = "resnet50", out_channels: int = 64,
                 latent_size: Tuple[int,int,int] = (128,128,10),
                 point_cloud_range: List[float] = ...,
                 depth_bins: int = 112, pretrained: bool = True) -> None: ...

    def forward(self, imgs: Tensor, cam_params: CameraParams) -> Tuple[Tensor, Tensor]:
        """
        imgs: (B, T, N_cam, 3, H_img, W_img)
        returns:
          voxel_feats: (B, T, out_channels, 128, 128, 10)
          depth_pred:  (B*T, N_cam, depth_bins, H_feat, W_feat)  # for optional depth loss
        """
```

`CameraParams` is a dataclass in `drift/models/encoders/camera_params.py` holding
`rots (B,T,N,3,3)`, `trans (B,T,N,3)`, `intrins (B,T,N,3,3)`, `post_rots (B,T,N,3,3)`,
`post_trans (B,T,N,3)`.

Implementation notes: ResNet+FPN → single-scale feature; predict a depth distribution over
`depth_bins`; outer-product with context features; splat into the latent voxel grid via
`bev_pool`-style cumulative-sum trick (a plain `scatter_add_` implementation is acceptable and
preferred for clarity — mark it `# TODO(perf)`).

### 2.2 `drift/models/encoders/lidar_encoder.py`

```python
class LidarEncoder(nn.Module):
    """LiDAR points -> latent 3D volume. MODIFIED from Doracamom (real 3D volume, no height broadcast)."""
    def __init__(self, in_channels: int = 4, out_channels: int = 64,
                 latent_size: Tuple[int,int,int] = (128,128,10),
                 point_cloud_range: List[float] = ...,
                 backbone: Literal["pillar3d","voxelnet"] = "pillar3d") -> None: ...

    def forward(self, points: List[List[Tensor]]) -> Tensor:
        """
        points: nested list [B][T] of (N_pts_i, in_channels) float tensors in that frame's LiDAR coords.
                Ragged by design — do not require padding.
        returns: (B, T, out_channels, 128, 128, 10)
        """
```

**Critical**: the output must have genuinely different features at different `z`. Do **not**
compute a BEV map and repeat it along height. Use a 3D voxelization (mean/max pooling of points
per voxel → small 3D conv stack), or pillar features followed by a learned `Conv2d(C -> C*Z)`
z-unfold. Prefer the former.

Implement voxelization with `torch.scatter_reduce` on flattened voxel indices — no custom CUDA.

### 2.3 `drift/models/encoders/voxel_query_generator.py`

```python
class CoarseVoxelQueryGenerator(nn.Module):
    """Initialize voxel queries from LiDAR geometry + camera semantics. MODIFIED from Doracamom CVQG."""
    def __init__(self, embed_dims: int = 128, latent_size=..., point_cloud_range=...,
                 fusion: Literal["sum","concat","gate"] = "gate") -> None: ...

    def forward(self, lidar_vol: Tensor, cam_vol: Tensor) -> Tensor:
        """
        lidar_vol: (B, T, C_l, X, Y, Z)
        cam_vol:   (B, T, C_c, X, Y, Z)
        returns queries: (B, T, embed_dims, X, Y, Z)
        """
```

Doracamom sums `Q_R + Q_I` because radar is a weak low-norm prior. With LiDAR that lets geometry
dominate uncontrolled. Default to `fusion="gate"`:
`Q = proj_l(Q_l) + sigmoid(g(Q_l)) * proj_c(Q_c)`. Keep `"sum"` and `"concat"` for ablation.

### 2.4 `drift/models/cmli.py` — ★ NOVEL

```python
class CrossModalLatentImagination(nn.Module):
    """
    ★ NOVEL. Reconstruct a missing/degraded modality's latent from the surviving modality plus
    temporal context, so forecasting degrades gracefully under sensor dropout.
    """
    def __init__(self, channels: int, latent_size=..., hidden: int = 64,
                 use_temporal_context: bool = True) -> None: ...

    def forward(self, lidar_vol: Optional[Tensor], cam_vol: Optional[Tensor],
                lidar_mask: Tensor, cam_mask: Tensor
               ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        lidar_vol, cam_vol: (B, T, C, X, Y, Z); a dropped modality may be None or zeros.
        lidar_mask, cam_mask: (B, T) float in {0,1}; 1 = modality present at that frame.
        returns (lidar_out, cam_out, aux) where aux contains
          'imagined_lidar', 'imagined_cam' (for the consistency loss) and
          'disagreement': (B, T, 1, X, Y, Z) cross-modal disagreement, fed to the uncertainty head.
        Where a mask is 1 the real latent passes through unchanged; where 0 the imagined latent
        is substituted. Both masks 0 at a frame is legal (pure temporal extrapolation).
        """
```

Generator: a small 3D conv encoder-decoder over the surviving modality, optionally concatenated
with the previous frame's fused latent. Must be cheap (≤ 5% of total FLOPs).

`ModalityDropout` (same file) is the **training-time** augmentation producing the masks:
per-sample, per-modality Bernoulli dropout with configurable rate, plus a "degrade" mode that
keeps a random subset of LiDAR points / darkens images rather than full removal.

### 2.5 `drift/models/encoders/cross_modal_fusion.py`

```python
class CrossModalFusion(nn.Module):
    """Two-sided gated fusion of LiDAR and camera latents. MODIFIED from Doracamom CMF."""
    def __init__(self, channels: int, out_channels: Optional[int] = None,
                 two_sided: bool = True, aux_heads: bool = True) -> None: ...

    def forward(self, lidar_vol: Tensor, cam_vol: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        returns fused (B,T,C_out,X,Y,Z) and aux dict with optional
          'occ_mask_logits': (B,T,1,X,Y,Z)  binary occupied/free auxiliary
        """
```

Doracamom's residual carries **only** the camera feature (`add_radar=False`), encoding
"camera primary, radar modulates". That asymmetry is wrong for LiDAR. Default `two_sided=True`:
```
f  = ConvBNReLU3D(concat(L, C))
out = f * sigmoid(g_l(f)) + f * sigmoid(g_c(f)) + L + C      # two-sided residual
```

### 2.6 `drift/models/taf.py`

```python
class TriplingAttentionFusion(nn.Module):
    """Tripling-Attention Fusion. REUSED from OccProphet (Sec. 3.2.2)."""
    def __init__(self, embed_dims: int, num_heads: int, window_size: int = 7,
                 timesteps: int = 3, height_kernel_size: int = 3, residual: bool = False) -> None: ...
    def forward(self, vox_feats: Tensor) -> Tensor:   # (B,T,C,X,Y,Z) -> (B,T,C,X,Y,Z)
```

Three branches over the latent volume, each followed by causal temporal attention, then combined
by broadcast addition:
- **scene**: `AdaptiveAvgPool3d(1)` → 1×1×1 conv → `(B,T,C)`
- **bev**: mean over z → windowed multi-head self-attention over `X*Y` tokens → `(B,T,X*Y,C)`
- **height**: mean over x,y → `Conv1d` over z → `(B,T,Z,C)`

Output `= bev + scene[:, :, None, :]` reshaped to `(B,T,C,X,Y,1)` **plus** height reshaped to
`(B,T,C,1,1,Z)`, broadcast-summed. Temporal attention is **causal** (lower-triangular mask).
A plain `nn.MultiheadAttention` with a shifted-window partition is fine; a full `ShiftWindowMSA`
is not required — mark simplifications in the docstring.

### 2.7 `drift/models/e4a.py`

```python
class EfficientAggregation4D(nn.Module):
    """Efficient 4D Aggregation (UNet over the latent volume). REUSED from OccProphet (Sec. 3.2.1)."""
    def __init__(self, in_channels: int, embed_dims: int, out_channels: Optional[int] = None,
                 downsample_layers: int = 3, in_proj_kernel_size: int = 1,
                 upsample_plugins: Optional[List[nn.Module]] = None,
                 timesteps: int = 3, multiscale_output: bool = False) -> None: ...
    def forward(self, vox_feats: Tensor) -> Tensor | List[Tensor]   # (B,T,C,X,Y,Z)
```

Symmetric 3D UNet applied per-timestep (fold `T` into batch for convs), with TAF plugins inserted
on the **upsample** path at each scale, and residual skip connections. `128→64→32→16` by default.
Channels double each downsample. This module is used **twice**: as the Observer body and as the
Refiner body.

### 2.8 `drift/models/observer.py`

```python
class Observer(nn.Module):
    """Aggregate multi-frame fused latents into a spacetime representation. REUSED (OccProphet)."""
    def __init__(self, in_channels: int, embed_dims: int, timesteps: int,
                 with_ego_channels: bool = True, **e4a_kwargs) -> None: ...
    def forward(self, fused: Tensor, ego_motion: Tensor) -> Tensor:
        """
        fused: (B, T_p, C, X, Y, Z)   already warped to the present frame
        ego_motion: (B, T_p, 4, 4)
        returns O_obs: (B, T_p, embed_dims, X, Y, Z)
        """
```

Converts each 4×4 to a 6-DoF vector (`mat2pose_vec`), broadcasts spatially, prepends a zero vector
for t=0, concatenates as 6 extra channels, then runs E4A. So `in_channels` seen by E4A is `C + 6`.

### 2.9 `drift/models/static_path.py` — part of ★ NOVEL forecaster

```python
class StaticForecastPath(nn.Module):
    """Forecast the static scene by ego-pose warping. Near parameter-free."""
    def __init__(self, latent_size=..., point_cloud_range=..., learned_residual: bool = True,
                 channels: Optional[int] = None) -> None: ...
    def forward(self, obs_latent: Tensor, future_ego: Tensor) -> Tensor:
        """
        obs_latent: (B, C, X, Y, Z) — the present-frame observation latent
        future_ego: (B, T_o, 4, 4) — cumulative transforms present -> each future frame
        returns: (B, T_o, C, X, Y, Z)
        """
```

Use **`grid_sample` with `align_corners=False`**, not the nearest-neighbour forward scatter used
in the reference code (that implementation also silently drops ~3% of voxels through an
off-by-one `range(split_num-1)`). Build a normalized ego-centred grid, apply the inverse transform,
sample. Optionally add a small learned residual conv.

### 2.10 `drift/models/instance_path.py` — ★ NOVEL (core contribution)

```python
@dataclass
class InstanceState:
    """Per-query agent state. All tensors (B, Q, ...)."""
    center: Tensor      # (B, Q, 3) metric xyz in present-frame ego coords
    size: Tensor        # (B, Q, 3) l,w,h
    yaw: Tensor         # (B, Q, 1)
    velocity: Tensor    # (B, Q, 3) m/s
    logits: Tensor      # (B, Q, num_classes)
    embed: Tensor       # (B, Q, C) query feature
    score: Tensor       # (B, Q, 1) objectness in [0,1]


class InstanceQueryExtractor(nn.Module):
    """★ NOVEL. Extract dynamic-agent queries from the observation latent."""
    def __init__(self, in_channels: int, embed_dims: int = 256, num_queries: int = 300,
                 num_classes: int = 2, num_decoder_layers: int = 3) -> None: ...
    def forward(self, obs_latent: Tensor) -> InstanceState:
        """obs_latent: (B, C, X, Y, Z) -> InstanceState with Q = num_queries"""


class MotionForecaster(nn.Module):
    """★ NOVEL. Roll instance queries forward to each future horizon."""
    def __init__(self, embed_dims: int = 256, num_future: int = 6,
                 mode: Literal["gru","transformer"] = "gru",
                 scene_condition_dims: Optional[int] = None) -> None: ...
    def forward(self, state: InstanceState, scene_cond: Optional[Tensor] = None
               ) -> List[InstanceState]:
        """returns a list of length num_future, each an InstanceState at that horizon"""


class InstanceSplatter(nn.Module):
    """★ NOVEL. Rasterize forecasted instance states back into the latent voxel grid."""
    def __init__(self, embed_dims: int, out_channels: int, latent_size=...,
                 point_cloud_range=..., soft: bool = True, sigma: float = 1.0) -> None: ...
    def forward(self, states: List[InstanceState]) -> Tuple[Tensor, Tensor]:
        """
        returns:
          dyn_feats: (B, T_o, out_channels, X, Y, Z)
          dyn_occ:   (B, T_o, 1, X, Y, Z)  soft occupancy mass, used as the merge gate
        """
```

**Splatting must be differentiable.** For each query, compute a soft box-indicator over voxels —
a Gaussian / smooth-box in the query's local (rotated) frame — weight by `score`, and
`scatter_add` the query embedding into the grid. Vectorize over queries; do **not** loop in
Python over `Q` for the whole grid (loop over queries only if you first restrict to each query's
local bounding sub-grid, which is the recommended approach and is fast).

### 2.11 `drift/models/condition.py`

```python
class ConditionalForecaster(nn.Module):
    """
    Condition Generator + Conditional Forecaster. REUSED from OccProphet (Sec. 3.3).
    A hypernetwork: predicts a Conv3d kernel from the global scene condition, then applies it
    grouped over the batch to map T_p input frames to T_o output frames.
    """
    def __init__(self, in_timesteps: int, out_timesteps: int, in_channels: int,
                 kernel_size: int = 1, norm_and_act: bool = True) -> None: ...
    def forward(self, vox_feats: Tensor) -> Tensor:   # (B,T_p,C,X,Y,Z) -> (B,T_o,C,X,Y,Z)
```

Mechanism (reproduce faithfully): `AdaptiveAvgPool3d(1)` + conv per frame → flatten to `(B, T_p*C)`
→ `Linear` predicting `T_o*C*T_p*C*k^3` weights → reshape to `(B*T_o*C, T_p*C, k,k,k)` →
`F.conv3d(x.reshape(1, B*T_p*C, X,Y,Z), kernel, groups=B)`. Guard the parameter count: with
`C=32, T_p=3, T_o=6, k=1` this Linear is ~1.8 M params — assert `in_timesteps*in_channels *
out_timesteps*in_channels*k**3 < 5e7` and raise a clear error otherwise.

### 2.12 `drift/models/forecaster.py` — ★ NOVEL assembly

```python
class DecoupledForecaster(nn.Module):
    """
    ★ NOVEL. Static scene by ego-warp + dynamic agents in instance-query space, merged by a
    learned gate. Contrast with DFIT-OccWorld, which decouples using a DENSE per-voxel flow field.
    """
    def __init__(self, channels: int, num_future: int, static_cfg: dict, instance_cfg: dict,
                 condition_cfg: Optional[dict] = None, merge: Literal["gate","sum"] = "gate") -> None: ...

    def forward(self, obs_latent: Tensor, future_ego: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """
        obs_latent: (B, T_p, C, X, Y, Z)
        future_ego: (B, T_o, 4, 4)
        returns future_latent (B, T_o, C, X, Y, Z) and aux containing
          'instance_states': List[InstanceState], 'dyn_occ', 'static_latent', 'dyn_latent',
          'present_state': InstanceState  (for the detection auxiliary loss)
        """
```

Merge: `out = static * (1 - a) + dyn * a` where `a = sigmoid(gate(concat(static, dyn, dyn_occ)))`.
The optional `ConditionalForecaster` provides scene-conditioned context to both paths.

### 2.13 `drift/models/refiner.py`

```python
class Refiner(nn.Module):
    """Reconcile static/dynamic seams via spatiotemporal interaction. REUSED (OccProphet Sec. 3.4)."""
    def __init__(self, channels: int, in_timesteps: int, out_timesteps: int, **e4a_kwargs) -> None: ...
    def forward(self, obs_latent: Tensor, future_latent: Tensor) -> Tensor:
        """concat along time -> E4A -> return only the future slice (B, T_o, C, X, Y, Z)"""
```

### 2.14 `drift/models/predictor.py`

```python
class OccupancyHead(nn.Module):
    """Per-voxel semantic logits at full resolution."""
    def forward(self, feats: Tensor) -> Tensor:   # (B,T_o,C,X,Y,Z) -> (B,T_o,num_classes,X,Y,Z)

class FlowHead(nn.Module):
    """Per-voxel 3D flow, in units of LATENT voxels (0.8 m), backward centroid-offset convention."""
    def forward(self, feats: Tensor) -> Tensor:   # -> (B,T_o,3,X,Y,Z)

class UncertaintyHead(nn.Module):
    """
    ★ NOVEL. Per-voxel predictive uncertainty that grows with horizon.
    Consumes the latent plus the CMLI cross-modal disagreement signal.
    """
    def __init__(self, in_channels: int, num_future: int, mode: Literal["variance","evidential"] = "variance",
                 use_disagreement: bool = True) -> None: ...
    def forward(self, feats: Tensor, disagreement: Optional[Tensor] = None) -> Tensor:
        """-> (B, T_o, 1, X, Y, Z), raw log-variance (unbounded); callers apply softplus/exp."""
```

Heads output at **latent resolution** `(128,128,10)`. Upsampling to `(512,512,40)` happens only
in the metric, by trilinear interpolation of **logits** followed by argmax — never argmax first.

### 2.15 `drift/models/drift.py`

```python
class DRIFT(nn.Module):
    """Top-level assembly. See docs/DESIGN_SPEC.md §2.15."""
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]: ...
    def loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]: ...
```

Pipeline order:
```
imgs, points
  -> CameraEncoder, LidarEncoder                       (B,T_p,C,X,Y,Z) each
  -> ModalityDropout (train) -> CMLI                   fills missing modalities
  -> CoarseVoxelQueryGenerator -> CrossModalFusion     fused (B,T_p,C,X,Y,Z)
  -> warp past frames into present frame (ego motion)
  -> Observer                                          O_obs (B,T_p,C',X,Y,Z)
  -> DecoupledForecaster                               (B,T_o,C',X,Y,Z)
  -> Refiner                                           (B,T_o,C',X,Y,Z)
  -> OccupancyHead / FlowHead / UncertaintyHead
```

`loss()` returns a dict of **scalar** tensors; the trainer sums them. Every key must start with
`loss_`.

---

## 3. Losses — `drift/losses/`

| File | Function | Notes |
|---|---|---|
| `occupancy.py` | `occupancy_loss(pred, target, class_weights, weights_cfg)` | CE + Lovász-softmax + `geo_scal` + `sem_scal`. Unlike the OccProphet release (which silently runs CE only and hard-codes weight 0.5), **all four terms are live and config-weighted here.** |
| `occupancy.py` | `downsample_target(target, ratio)` | 512→128 by masked mode vote; all-empty block → 0; non-empty block with no majority → **255 (ignore)** |
| `flow.py` | `flow_loss(pred, target)` | SmoothL1, mask = all 3 channels != 255, NaN-guard returning a 0 with valid graph when nothing is kept |
| `instance.py` | `instance_loss(states, gt_boxes)` | Hungarian matching (focal cls + L1 box), plus a trajectory/ADE term over forecasted horizons |
| `cmli.py` | `cmli_consistency_loss(aux, masks)` | L2 between imagined and real latent on frames where the modality **was** present (teacher signal), + an occupancy-consistency term |
| `uncertainty.py` | `uncertainty_nll(logits, log_var, target)` | Heteroscedastic NLL; must not collapse (add the standard `0.5*log_var` regularizer) |

Class weights default: `w[0] = 1.0` (free), `w[1:] = 5.0`.

---

## 4. Data — `drift/data/`

`cam4docc_dataset.py` implements a **protocol-compatible** dataset. It must reproduce the sample
contract below; the on-disk generation path may be a separate offline script.

```python
batch = {
  "imgs":            (B, T_p, N_cam, 3, H, W)   float32
  "cam_params":      CameraParams                # each (B, T_p, N_cam, ...)
  "points":          List[List[Tensor]]          # [B][T_p] ragged (N_i, 4)
  "ego_motion":      (B, T_seq, 4, 4)            # T_{t+1 <- t}
  "future_ego":      (B, T_o, 4, 4)              # cumulative present -> future t (derived)
  "gt_occ":          (B, T_o, 512, 512, 40)      int64
  "gt_flow":         (B, T_o, 3, 128, 128, 10)   float32, 255 = ignore
  "gt_instance":     (B, T_o, 512, 512, 40)      int64   (0 = bg)
  "gt_boxes":        List[List[BoxSet]]          # for the instance auxiliary loss
  "lidar_mask":      (B, T_p) float32            # 1 = present
  "cam_mask":        (B, T_p) float32
}
```

Notes on fidelity to the reference protocol (implement these, they are not bugs to fix silently):
- GT is rasterized in the **present** frame for all timesteps, not per-frame frames.
- Instances first appearing in the future are excluded.
- Objects whose OBB is not fully inside `pc_range` are dropped for that frame.
- Flow GT units are latent voxels (0.8 m), backward, 255 = ignore.

`ego_motion.py` provides `mat2pose_vec`, `cumulative_warp_to_present`, `compose_future_transforms`.

---

## 5. Metrics — `drift/metrics/`

`iou.py` must be **numerically identical** to the reference protocol:
- trilinear-upsample logits `(num_cls,128,128,10) -> (num_cls,512,512,40)`, then argmax
- `fast_hist(pred, label, num_cls)` = `bincount(num_cls*label + pred).reshape(num_cls, num_cls)`
- accumulate **one pooled confusion matrix** across all frames and all samples; compute IoU once
  from the pooled matrix (this is a micro-average over voxels, macro over classes — *not* a mean
  of per-sample IoUs)
- `IOU_mean = mean(ious[1:])` — class 0 (free) excluded
- `IoU_c` = present frame only; `IoU_f` = all future frames pooled
- per-horizon buckets are **cumulative** (bucket *k* pools frames up to horizon *k*)

`flow_epe.py`: EPE, angular error, magnitude error, all masked by `!= 255`, reported in metres
(multiply latent-voxel units by 0.8).

`calibration.py`: ECE with configurable bins, plus reliability-curve data. ★
`robustness.py`: run the model under a grid of dropout scenarios and emit the degradation table. ★

---

## 6. Configuration

Plain Python dataclasses in `configs/` (no mmcv dependency). `configs/base.py` defines
`DriftConfig` with nested `ModelConfig`, `DataConfig`, `TrainConfig`, `LossConfig`.
`configs/drift_fusion_nuscenes.py` instantiates the default. Every hyperparameter named in this
spec must be reachable from the config. Provide `cam4docc_2s` and `extended_3s` presets, and
ablation presets: `no_cmli`, `no_instance_path` (dense-flow fallback), `no_uncertainty`,
`camera_only`, `lidar_only`, `fusion_sum`.

---

## 7. Engineering requirements

- **Python 3.10+, PyTorch 2.x.** No mmcv / mmdet3d / spconv **hard** dependency — the model must
  import and run on plain PyTorch (CPU included, small grid) so it is testable. Optional
  accelerated paths may be behind `try: import ...`.
- One module per file. Full type hints. Google-style docstrings.
- Every module's docstring states REUSED / MODIFIED / NOVEL and cites the origin paper.
- Multi-GPU via `torch.nn.parallel.DistributedDataParallel` in `tools/train.py`.
- Determinism helper in `drift/utils/seed.py`.
- No `assert` for user-facing validation — raise `ValueError` with an actionable message.
- Never use `Date.now()`-style nondeterminism in module init.
- **Smoke-testability is mandatory**: every module must run with a tiny config
  (`latent_size=(16,16,4)`, `B=1, T_p=2, T_o=3, num_queries=8`) on CPU.

## 8. Tests — `tests/`

`test_shapes.py` — every module's output shape matches this spec, at the tiny config.
`test_forward_backward.py` — full `DRIFT` forward + `loss()` + `.backward()` on synthetic data;
assert every registered parameter that requires grad receives a non-None grad (catches dead
branches — in particular the instance path must receive gradient through the splatter).
`test_metrics.py` — pooled-IoU against a hand-computed confusion matrix; flow EPE masking.
`test_warp.py` — identity ego transform must be a no-op; a pure translation must shift the volume
by the expected number of voxels.
