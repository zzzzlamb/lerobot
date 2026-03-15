#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def validate_centered_window_size(window_size: int, *, field_name: str = "window_size") -> None:
    if window_size < 1:
        raise ValueError(f"{field_name} must be >= 1, got {window_size}.")
    if window_size > 1 and window_size % 2 == 0:
        raise ValueError(f"{field_name} must be odd when smoothing is enabled, got {window_size}.")


def normalize_excluded_indices(action_dim: int, excluded_indices: Sequence[int] | None = None) -> list[int]:
    normalized = set()
    for index in excluded_indices or []:
        resolved = index + action_dim if index < 0 else index
        if resolved < 0 or resolved >= action_dim:
            raise ValueError(
                f"Excluded action index {index} resolves to {resolved}, outside valid range [0, {action_dim})."
            )
        normalized.add(resolved)
    return sorted(normalized)


def get_smoothed_action_indices(action_dim: int, excluded_indices: Sequence[int] | None = None) -> list[int]:
    excluded = set(normalize_excluded_indices(action_dim, excluded_indices))
    return [index for index in range(action_dim) if index not in excluded]


def _build_numpy_left_pad(values: np.ndarray, pad_left: int, left_context: np.ndarray | None) -> np.ndarray:
    if pad_left == 0:
        return np.empty((0, values.shape[1]), dtype=values.dtype)

    if left_context is None or left_context.size == 0:
        return np.repeat(values[:1], pad_left, axis=0)

    context = left_context[-pad_left:]
    if len(context) == pad_left:
        return context.astype(values.dtype, copy=False)

    prepend = np.repeat(context[:1], pad_left - len(context), axis=0)
    return np.concatenate([prepend, context], axis=0).astype(values.dtype, copy=False)


def smooth_action_array(
    values: np.ndarray,
    window_size: int,
    *,
    excluded_indices: Sequence[int] | None = None,
    left_context: np.ndarray | None = None,
) -> np.ndarray:
    validate_centered_window_size(window_size)
    if values.ndim != 2:
        raise ValueError(f"Expected action array with shape (timesteps, action_dim), got {values.shape}.")
    if window_size == 1 or len(values) == 0:
        return values.copy()

    action_dim = values.shape[1]
    smoothed_indices = get_smoothed_action_indices(action_dim, excluded_indices)
    if not smoothed_indices:
        return values.copy()

    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    kernel = np.ones(window_size, dtype=np.float32) / window_size

    smoothed = values.astype(np.float32, copy=True)
    left_pad = _build_numpy_left_pad(values, pad_left, left_context)
    right_pad = np.repeat(values[-1:], pad_right, axis=0)
    padded = np.concatenate([left_pad, values, right_pad], axis=0)

    for action_index in smoothed_indices:
        smoothed[:, action_index] = np.convolve(padded[:, action_index], kernel, mode="valid")

    return smoothed.astype(values.dtype, copy=False)


def _ensure_3d_actions(actions: Tensor) -> tuple[Tensor, bool]:
    if actions.ndim == 1:
        return actions.unsqueeze(0).unsqueeze(0), True
    if actions.ndim == 2:
        return actions.unsqueeze(0), True
    if actions.ndim == 3:
        return actions, False
    raise ValueError(
        "Expected action tensor with shape (batch, timesteps, action_dim), "
        f"(timesteps, action_dim), or (action_dim,), got {tuple(actions.shape)}."
    )


def _build_torch_left_pad(actions_bta: Tensor, pad_left: int, left_context: Tensor | None) -> Tensor:
    if pad_left == 0:
        return actions_bta[:, :0]

    if left_context is None or left_context.numel() == 0:
        return actions_bta[:, :1].expand(-1, pad_left, -1)

    context_bta, _ = _ensure_3d_actions(left_context)
    context_tail = context_bta[:, -pad_left:]
    if context_tail.shape[1] == pad_left:
        return context_tail.to(device=actions_bta.device, dtype=actions_bta.dtype)

    prepend = context_tail[:, :1].expand(-1, pad_left - context_tail.shape[1], -1)
    return torch.cat([prepend, context_tail.to(device=actions_bta.device, dtype=actions_bta.dtype)], dim=1)


def smooth_action_tensor(
    actions: Tensor,
    window_size: int,
    *,
    excluded_indices: Sequence[int] | None = None,
    left_context: Tensor | None = None,
) -> Tensor:
    validate_centered_window_size(window_size)
    actions_bta, squeeze_batch = _ensure_3d_actions(actions)
    if window_size == 1 or actions_bta.shape[1] == 0:
        return actions.clone()

    action_dim = actions_bta.shape[-1]
    smoothed_indices = get_smoothed_action_indices(action_dim, excluded_indices)
    if not smoothed_indices:
        return actions.clone()

    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    smoothed = actions_bta.clone()

    left_pad = _build_torch_left_pad(actions_bta, pad_left, left_context)
    right_pad = actions_bta[:, -1:].expand(-1, pad_right, -1) if pad_right > 0 else actions_bta[:, :0]
    padded = torch.cat([left_pad, actions_bta, right_pad], dim=1).transpose(1, 2)

    kernel = torch.ones(
        len(smoothed_indices),
        1,
        window_size,
        device=actions_bta.device,
        dtype=actions_bta.dtype,
    ) / window_size
    filtered = F.conv1d(padded[:, smoothed_indices], kernel, groups=len(smoothed_indices))
    smoothed[:, :, smoothed_indices] = filtered.transpose(1, 2)

    return smoothed.squeeze(0) if squeeze_batch else smoothed


def generate_transition_actions(
    last_action: Tensor,
    next_action: Tensor,
    num_transition_steps: int,
    *,
    excluded_indices: Sequence[int] | None = None,
) -> Tensor | None:
    if num_transition_steps <= 0:
        return None

    if last_action.ndim == 1:
        last_action = last_action.unsqueeze(0)
    if next_action.ndim == 1:
        next_action = next_action.unsqueeze(0)
    if last_action.ndim == 2:
        last_action = last_action.unsqueeze(1)
    if next_action.ndim == 2:
        next_action = next_action.unsqueeze(1)

    last_action_ba, squeeze_batch = _ensure_3d_actions(last_action)
    next_action_ba, _ = _ensure_3d_actions(next_action)
    last_action_ba = last_action_ba[:, 0]
    next_action_ba = next_action_ba[:, 0]

    action_dim = last_action_ba.shape[-1]
    smoothed_indices = get_smoothed_action_indices(action_dim, excluded_indices)
    if not smoothed_indices:
        return None

    transition_actions = []
    for step in range(1, num_transition_steps + 1):
        alpha = step / (num_transition_steps + 1)
        interpolated = next_action_ba.clone()
        interpolated[:, smoothed_indices] = (
            last_action_ba[:, smoothed_indices]
            + alpha * (next_action_ba[:, smoothed_indices] - last_action_ba[:, smoothed_indices])
        )
        transition_actions.append(interpolated)

    stacked = torch.stack(transition_actions, dim=1)
    return stacked.squeeze(0) if squeeze_batch else stacked


def compute_smoothness_loss(
    actions: Tensor,
    window_size: int,
    *,
    excluded_indices: Sequence[int] | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    validate_centered_window_size(window_size)
    if window_size == 1:
        return actions.new_zeros(())

    actions_bta, _ = _ensure_3d_actions(actions)
    action_dim = actions_bta.shape[-1]
    smoothed_indices = get_smoothed_action_indices(action_dim, excluded_indices)
    if not smoothed_indices:
        return actions.new_zeros(())

    if mask is None:
        smoothed = smooth_action_tensor(actions_bta, window_size, excluded_indices=excluded_indices)
        diff = torch.abs(actions_bta[..., smoothed_indices] - smoothed[..., smoothed_indices])
        return diff.mean()

    mask_bt = mask
    if mask_bt.ndim == 1:
        mask_bt = mask_bt.unsqueeze(0)
    if mask_bt.ndim != 2:
        raise ValueError(f"Expected mask with shape (batch, timesteps) or (timesteps,), got {tuple(mask.shape)}.")

    mask_bt = mask_bt.to(device=actions_bta.device, dtype=torch.bool)
    smoothed = actions_bta.clone()
    for batch_index in range(actions_bta.shape[0]):
        valid_len = int(mask_bt[batch_index].sum().item())
        if valid_len == 0:
            continue
        smoothed[batch_index, :valid_len] = smooth_action_tensor(
            actions_bta[batch_index, :valid_len],
            window_size,
            excluded_indices=excluded_indices,
        )

    diff = torch.abs(actions_bta[..., smoothed_indices] - smoothed[..., smoothed_indices])
    valid_mask = mask_bt.unsqueeze(-1).to(device=diff.device, dtype=diff.dtype).expand_as(diff)
    valid_count = valid_mask.sum().clamp_min(1.0)
    return (diff * valid_mask).sum() / valid_count
