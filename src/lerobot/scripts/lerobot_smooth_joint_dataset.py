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

"""
Offline joint smoothing for LeRobot v3 datasets.

This script creates a copy of an existing dataset, applies either centered
moving-average smoothing or the legacy vkrobot mean filter on a per-episode
basis, and updates the feature statistics stored in `meta/stats.json` and
`meta/episodes/*.parquet`.

Examples:

    PYTHONPATH=src python src/lerobot/scripts/lerobot_smooth_joint_dataset.py \
      --repo-id zzzlamb/so100_test2_clean \
      --root /home/zyb/.cache/huggingface/lerobot/zzzlamb/so100_test2_clean \
      --output-repo-id zzzlamb/so100_test2_clean_smooth5 \
      --output-root /home/zyb/.cache/huggingface/lerobot/zzzlamb/so100_test2_clean_smooth5 \
      --feature-keys action \
      --window-size 5

    PYTHONPATH=src python src/lerobot/scripts/lerobot_smooth_joint_dataset.py \
      --repo-id zzzlamb/so100_test2_clean \
      --root /home/zyb/.cache/huggingface/lerobot/zzzlamb/so100_test2_clean \
      --output-repo-id zzzlamb/so100_test2_clean_smooth5_both \
      --output-root /home/zyb/.cache/huggingface/lerobot/zzzlamb/so100_test2_clean_smooth5_both \
      --feature-keys action observation.state \
      --window-size 5 \
      --include-gripper
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_stats, write_stats
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth joint trajectories in a LeRobot dataset copy.")
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Source dataset root. Defaults to HF_LEROBOT_HOME/repo_id.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output repo id. Defaults to '<repo-id>_smooth_w<window-size>'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to HF_LEROBOT_HOME/output_repo_id.",
    )
    parser.add_argument(
        "--feature-keys",
        nargs="+",
        default=["action"],
        choices=["action", "observation.state"],
        help="Feature keys to smooth. Default: action.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Odd moving-average window size. Default: 5.",
    )
    parser.add_argument(
        "--mode",
        choices=["centered", "legacy_vkrobot"],
        default="centered",
        help="Smoothing mode. 'legacy_vkrobot' exactly reproduces the external dataset smoothing logic.",
    )
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Also smooth joints whose name contains 'gripper'. Disabled by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output_root if it already exists.",
    )
    return parser.parse_args()


def centered_moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}.")
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (frames, dims), got {values.shape}.")

    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    smoothed = np.empty_like(values, dtype=np.float32)

    for dim_idx in range(values.shape[1]):
        smoothed[:, dim_idx] = np.convolve(padded[:, dim_idx], kernel, mode="valid")

    return smoothed


def legacy_vkrobot_mean_filter(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (frames, dims), got {values.shape}.")
    if values.shape[1] != 6:
        raise ValueError(
            "legacy_vkrobot smoothing expects exactly 6 action dimensions, "
            f"got {values.shape[1]}."
        )

    mean_num = 5
    smoothed = values.astype(np.float32, copy=True)

    for frame_index in range(values.shape[0]):
        if frame_index < mean_num:
            continue
        if frame_index > values.shape[0] - mean_num - 1:
            continue

        for dim_index in range(values.shape[1]):
            total = 0.0
            for offset in range(1, mean_num + 1):
                total += values[frame_index + offset, dim_index]
                total += values[frame_index - offset, dim_index]
            smoothed[frame_index, dim_index] = total / (mean_num * 2.0)

    return smoothed.astype(values.dtype, copy=False)


def smooth_selected_joints(values: np.ndarray, joint_indices: list[int], window_size: int) -> np.ndarray:
    smoothed = values.copy().astype(np.float32, copy=False)
    if not joint_indices or window_size <= 1:
        return smoothed
    smoothed[:, joint_indices] = centered_moving_average(smoothed[:, joint_indices], window_size)
    return smoothed


def legacy_vkrobot_smooth_selected_joints(values: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    if values.shape[1] != len(joint_indices):
        raise ValueError(
            "legacy_vkrobot smoothing requires all action dimensions to be selected. "
            f"Got {len(joint_indices)} selected dims for shape {values.shape}."
        )
    return legacy_vkrobot_mean_filter(values)


def compute_motion_metrics(values: np.ndarray) -> dict[str, float]:
    if len(values) < 3:
        return {
            "delta_abs_mean": 0.0,
            "jerk_abs_mean": 0.0,
        }

    delta = np.diff(values, axis=0)
    jerk = np.diff(delta, axis=0)
    return {
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "jerk_abs_mean": float(np.mean(np.abs(jerk))),
    }


def summarize_metrics(feature_key: str, rows: list[dict]) -> None:
    if not rows:
        return

    before_delta = np.mean([row["before_delta_abs_mean"] for row in rows])
    after_delta = np.mean([row["after_delta_abs_mean"] for row in rows])
    before_jerk = np.mean([row["before_jerk_abs_mean"] for row in rows])
    after_jerk = np.mean([row["after_jerk_abs_mean"] for row in rows])
    deviation = np.mean([row["mean_abs_deviation"] for row in rows])

    logging.info(
        "%s: mean|delta| %.4f -> %.4f, mean|jerk| %.4f -> %.4f, mean abs deviation %.4f",
        feature_key,
        before_delta,
        after_delta,
        before_jerk,
        after_jerk,
        deviation,
    )


def resolve_root(repo_id: str, root: Path | None) -> Path:
    if root is not None:
        return root.expanduser().resolve()
    return (HF_LEROBOT_HOME / repo_id).expanduser().resolve()


def resolve_output(repo_id: str, output_repo_id: str | None, output_root: Path | None, window_size: int) -> tuple[str, Path]:
    final_repo_id = output_repo_id or f"{repo_id}_smooth_w{window_size}"
    if output_root is not None:
        return final_repo_id, output_root.expanduser().resolve()
    return final_repo_id, (HF_LEROBOT_HOME / final_repo_id).expanduser().resolve()


def resolve_joint_indices(feature_info: dict, include_gripper: bool) -> tuple[list[int], list[str]]:
    joint_names = feature_info.get("names")
    if not joint_names:
        raise ValueError("Selected feature does not define joint names.")

    indices = []
    selected_names = []
    for idx, joint_name in enumerate(joint_names):
        is_gripper = "gripper" in joint_name.lower()
        if is_gripper and not include_gripper:
            continue
        indices.append(idx)
        selected_names.append(joint_name)

    if not indices:
        raise ValueError("No joint indices selected. Use --include-gripper if needed.")

    return indices, selected_names


def update_episode_stats_file(episodes_path: Path, updated_episode_stats: dict[int, dict[str, dict]]) -> None:
    episodes_df = pd.read_parquet(episodes_path)

    for row_idx, episode_index in episodes_df["episode_index"].items():
        feature_stats = updated_episode_stats.get(int(episode_index))
        if feature_stats is None:
            continue

        for feature_key, stats in feature_stats.items():
            for stat_name, stat_value in stats.items():
                episodes_df.at[row_idx, f"stats/{feature_key}/{stat_name}"] = stat_value

    write_episode_dataframe(episodes_df, episodes_path)


def write_episode_dataframe(episodes_df: pd.DataFrame, episodes_path: Path) -> None:
    pydict = {}
    for column in episodes_df.columns:
        values = []
        for value in episodes_df[column].tolist():
            if isinstance(value, np.ndarray):
                values.append(value.tolist())
            else:
                values.append(value)
        pydict[column] = values

    table = pa.Table.from_pydict(pydict)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, episodes_path, compression="snappy", use_dictionary=True)


def main() -> None:
    args = parse_args()

    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1.")
    if args.mode == "centered" and args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd for centered smoothing.")
    if args.mode == "legacy_vkrobot" and args.feature_keys != ["action"]:
        raise ValueError("legacy_vkrobot mode only supports smoothing the 'action' feature.")
    if args.mode == "legacy_vkrobot" and not args.include_gripper:
        logging.warning("legacy_vkrobot mode always smooths all 6 action dimensions, including gripper.")

    init_logging()

    src_root = resolve_root(args.repo_id, args.root)
    output_repo_id, output_root = resolve_output(
        args.repo_id, args.output_repo_id, args.output_root, args.window_size
    )

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")
    if src_root == output_root:
        raise ValueError("output_root must be different from the source dataset root.")

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to replace it.")
        shutil.rmtree(output_root)

    logging.info("Copying dataset to %s", output_root)
    shutil.copytree(src_root, output_root)

    meta = LeRobotDatasetMetadata(output_repo_id, root=output_root)
    existing_stats = load_stats(output_root)
    if existing_stats is None:
        raise FileNotFoundError(f"Missing stats file in copied dataset: {output_root / 'meta' / 'stats.json'}")

    joint_indices_by_feature: dict[str, list[int]] = {}
    metrics_by_feature: dict[str, list[dict]] = {feature_key: [] for feature_key in args.feature_keys}

    for feature_key in args.feature_keys:
        if feature_key not in meta.features:
            raise KeyError(f"Feature '{feature_key}' not found in dataset.")
        if args.mode == "legacy_vkrobot":
            feature_info = meta.features[feature_key]
            indices = list(range(feature_info["shape"][0]))
            selected_names = feature_info.get("names") or [f"joint_{idx}" for idx in indices]
        else:
            indices, selected_names = resolve_joint_indices(meta.features[feature_key], args.include_gripper)
        joint_indices_by_feature[feature_key] = indices
        logging.info("Smoothing %s joints for %s: %s", len(indices), feature_key, ", ".join(selected_names))

    updated_episode_stats: dict[int, dict[str, dict]] = {}
    data_files = sorted((output_root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {output_root / 'data'}")

    for parquet_path in tqdm(data_files, desc="Smoothing data files"):
        df = pd.read_parquet(parquet_path)

        for episode_index, episode_rows in df.groupby("episode_index", sort=False):
            ordered_episode = episode_rows.sort_values("frame_index")
            ordered_indices = ordered_episode.index.to_list()
            updated_episode_stats.setdefault(int(episode_index), {})

            for feature_key in args.feature_keys:
                original = np.stack(ordered_episode[feature_key].to_numpy()).astype(np.float32)
                if args.mode == "legacy_vkrobot":
                    smoothed = legacy_vkrobot_smooth_selected_joints(
                        original, joint_indices_by_feature[feature_key]
                    )
                else:
                    smoothed = smooth_selected_joints(
                        original, joint_indices_by_feature[feature_key], args.window_size
                    )

                selected_original = original[:, joint_indices_by_feature[feature_key]]
                selected_smoothed = smoothed[:, joint_indices_by_feature[feature_key]]

                before_metrics = compute_motion_metrics(selected_original)
                after_metrics = compute_motion_metrics(selected_smoothed)
                metrics_by_feature[feature_key].append(
                    {
                        "before_delta_abs_mean": before_metrics["delta_abs_mean"],
                        "after_delta_abs_mean": after_metrics["delta_abs_mean"],
                        "before_jerk_abs_mean": before_metrics["jerk_abs_mean"],
                        "after_jerk_abs_mean": after_metrics["jerk_abs_mean"],
                        "mean_abs_deviation": float(np.mean(np.abs(selected_smoothed - selected_original))),
                    }
                )

                df.loc[ordered_indices, feature_key] = pd.Series(
                    [row for row in smoothed], index=ordered_indices, dtype=object
                )
                updated_episode_stats[int(episode_index)][feature_key] = get_feature_stats(
                    smoothed, axis=0, keepdims=False
                )

        _write_parquet(df.reset_index(drop=True), parquet_path, meta)

    episode_files = sorted((output_root / "meta" / "episodes").glob("*/*.parquet"))
    for episodes_path in tqdm(episode_files, desc="Updating episode stats"):
        update_episode_stats_file(episodes_path, updated_episode_stats)

    aggregated_feature_stats = aggregate_stats(list(updated_episode_stats.values()))
    for feature_key in args.feature_keys:
        existing_stats[feature_key] = aggregated_feature_stats[feature_key]
    write_stats(existing_stats, output_root)

    logging.info("Finished smoothing dataset copy.")
    logging.info("Source root: %s", src_root)
    logging.info("Output root: %s", output_root)
    for feature_key, rows in metrics_by_feature.items():
        summarize_metrics(feature_key, rows)


if __name__ == "__main__":
    main()
