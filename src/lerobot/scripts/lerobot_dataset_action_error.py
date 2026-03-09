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
Offline action error analysis on a dataset.

This script replays observations from a LeRobot dataset through a policy and compares predicted actions
against recorded ground-truth actions.

Example:
    python -m lerobot.scripts.lerobot_dataset_action_error \
        --policy.path=/path/to/pretrained_model \
        --dataset.repo_id=<USER>/so100_dataset \
        --dataset.root=~/.cache/huggingface/lerobot \
        --num_samples=500 \
        --batch_size=1 \
        --policy.device=cuda \
        --output_csv=outputs/action_error/per_sample.csv
"""

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class DatasetActionErrorConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    batch_size: int = 1
    num_workers: int = 0
    num_samples: int | None = None
    shuffle: bool = False
    seed: int = 0
    tolerance_s: float = 1e-4
    # If True, reset policy state (e.g. action queues) every batch for independent sample comparison.
    reset_policy_each_batch: bool = True
    output_csv: str | None = None
    output_metrics_json: str | None = None

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Policy path is required (--policy.path=...).")

        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _extract_observation_batch(
    batch: dict[str, torch.Tensor | list[str]],
    input_feature_keys: list[str],
) -> dict[str, torch.Tensor | list[str]]:
    missing = [k for k in input_feature_keys if k not in batch]
    if missing:
        raise ValueError(
            f"Dataset is missing policy input features: {missing}. "
            "Please verify dataset/policy compatibility."
        )

    obs = {k: batch[k] for k in input_feature_keys}
    if "task" in batch:
        obs["task"] = batch["task"]
    if "robot_type" in batch:
        obs["robot_type"] = batch["robot_type"]
    return obs


def _select_ground_truth_action(
    batch: dict[str, torch.Tensor | list[str]],
    action_delta_indices: list[int] | None,
) -> torch.Tensor:
    target = batch[ACTION]
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected tensor for '{ACTION}', got {type(target)}")

    if target.ndim == 2:
        # (B, action_dim)
        return target

    if target.ndim == 3:
        # (B, T, action_dim), choose timestep aligned with delta=0 when available.
        if action_delta_indices and len(action_delta_indices) == target.shape[1] and 0 in action_delta_indices:
            t0 = action_delta_indices.index(0)
            return target[:, t0, :]

        logging.warning(
            "Action tensor is 3D but couldn't locate delta=0 index from policy config; using timestep 0."
        )
        return target[:, 0, :]

    raise ValueError(
        f"Unsupported action tensor shape {tuple(target.shape)}. Expected (B, D) or (B, T, D)."
    )


def _action_names(dataset: LeRobotDataset, action_dim: int) -> list[str]:
    names = dataset.meta.features[ACTION].get("names", [])
    if isinstance(names, list) and len(names) == action_dim:
        return [str(n) for n in names]
    return [f"action_{i}" for i in range(action_dim)]


@parser.wrap()
def dataset_action_error(cfg: DatasetActionErrorConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.policy is None:
        raise ValueError("Policy configuration was not initialized.")

    torch.manual_seed(cfg.seed)

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        tolerance_s=cfg.tolerance_s,
    )
    logging.info(f"Loaded dataset with {len(dataset)} frames ({dataset.num_episodes} episodes).")

    generator = None
    if cfg.shuffle:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        generator=generator,
    )

    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
    )

    total_target = len(dataset) if cfg.num_samples is None else min(cfg.num_samples, len(dataset))
    processed = 0
    sum_abs_err: torch.Tensor | None = None
    sum_sq_err: torch.Tensor | None = None
    global_max_abs = 0.0
    action_names: list[str] | None = None

    csv_file = None
    csv_writer = None
    if cfg.output_csv:
        csv_path = Path(cfg.output_csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")

    try:
        with torch.inference_mode():
            progress = tqdm(total=total_target, desc="Comparing actions", unit="frame")
            for batch in dataloader:
                if processed >= total_target:
                    break

                if cfg.reset_policy_each_batch:
                    policy.reset()

                obs_batch = _extract_observation_batch(batch, list(cfg.policy.input_features.keys()))
                obs_batch = preprocessor(obs_batch)

                pred_action = policy.select_action(obs_batch)
                pred_action = postprocessor(pred_action)

                if pred_action.ndim == 3:
                    pred_action = pred_action[:, 0, :]
                if pred_action.ndim != 2:
                    raise ValueError(
                        f"Unsupported predicted action shape {tuple(pred_action.shape)}. Expected (B, D) or (B, T, D)."
                    )

                target_action = _select_ground_truth_action(batch, cfg.policy.action_delta_indices)

                pred_action = pred_action.detach().to("cpu", dtype=torch.float32)
                target_action = target_action.detach().to("cpu", dtype=torch.float32)

                if pred_action.shape[-1] != target_action.shape[-1]:
                    raise ValueError(
                        f"Action dimension mismatch: pred={pred_action.shape[-1]}, gt={target_action.shape[-1]}."
                    )

                remaining = total_target - processed
                if pred_action.shape[0] > remaining:
                    pred_action = pred_action[:remaining]
                    target_action = target_action[:remaining]

                err = pred_action - target_action
                abs_err = err.abs()
                sq_err = err.pow(2)

                if sum_abs_err is None:
                    sum_abs_err = torch.zeros(abs_err.shape[-1], dtype=torch.float64)
                    sum_sq_err = torch.zeros(abs_err.shape[-1], dtype=torch.float64)
                    action_names = _action_names(dataset, abs_err.shape[-1])

                sum_abs_err += abs_err.sum(dim=0, dtype=torch.float64)
                sum_sq_err += sq_err.sum(dim=0, dtype=torch.float64)
                global_max_abs = max(global_max_abs, abs_err.max().item())

                batch_size_eff = pred_action.shape[0]
                if csv_file:
                    if csv_writer is None:
                        assert action_names is not None
                        fieldnames = ["index", "episode_index", "mae", "rmse", "max_abs"]
                        for name in action_names:
                            fieldnames.extend([f"pred::{name}", f"gt::{name}", f"abs_err::{name}"])
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        csv_writer.writeheader()

                    indices = batch.get("index")
                    if isinstance(indices, torch.Tensor):
                        indices_list = indices[:batch_size_eff].tolist()
                    else:
                        indices_list = [None] * batch_size_eff

                    ep_indices = batch.get("episode_index")
                    if isinstance(ep_indices, torch.Tensor):
                        ep_list = ep_indices[:batch_size_eff].tolist()
                    else:
                        ep_list = [None] * batch_size_eff

                    for i in range(batch_size_eff):
                        row = {
                            "index": int(indices_list[i]) if indices_list[i] is not None else "",
                            "episode_index": int(ep_list[i]) if ep_list[i] is not None else "",
                            "mae": float(abs_err[i].mean().item()),
                            "rmse": float(torch.sqrt(sq_err[i].mean()).item()),
                            "max_abs": float(abs_err[i].max().item()),
                        }
                        assert action_names is not None
                        for dim, name in enumerate(action_names):
                            row[f"pred::{name}"] = float(pred_action[i, dim].item())
                            row[f"gt::{name}"] = float(target_action[i, dim].item())
                            row[f"abs_err::{name}"] = float(abs_err[i, dim].item())
                        csv_writer.writerow(row)

                processed += batch_size_eff
                progress.update(batch_size_eff)
                running_mae = float(sum_abs_err.sum().item() / (processed * sum_abs_err.numel()))
                progress.set_postfix(mae=f"{running_mae:.6f}")

            progress.close()

        if sum_abs_err is None or sum_sq_err is None or action_names is None:
            raise RuntimeError("No samples were processed. Check dataset / num_samples settings.")

        mae_per_dim = (sum_abs_err / processed).to(torch.float64)
        rmse_per_dim = torch.sqrt(sum_sq_err / processed)
        overall_mae = float(mae_per_dim.mean().item())
        overall_rmse = float(torch.sqrt(sum_sq_err.mean() / processed).item())

        logging.info("=" * 80)
        logging.info("Offline action error summary")
        logging.info(f"Samples compared: {processed}")
        logging.info(f"Overall MAE:  {overall_mae:.6f}")
        logging.info(f"Overall RMSE: {overall_rmse:.6f}")
        logging.info(f"Global max abs error: {global_max_abs:.6f}")
        logging.info("-" * 80)
        for name, mae, rmse in zip(action_names, mae_per_dim.tolist(), rmse_per_dim.tolist(), strict=True):
            logging.info(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")

        if cfg.output_metrics_json:
            metrics_path = Path(cfg.output_metrics_json).expanduser()
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "num_samples": processed,
                "overall_mae": overall_mae,
                "overall_rmse": overall_rmse,
                "global_max_abs_error": global_max_abs,
                "per_dim": {
                    name: {
                        "mae": float(mae),
                        "rmse": float(rmse),
                    }
                    for name, mae, rmse in zip(action_names, mae_per_dim.tolist(), rmse_per_dim.tolist(), strict=True)
                },
            }
            metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info(f"Saved metrics JSON to {metrics_path}")

        if cfg.output_csv:
            logging.info(f"Saved per-sample comparison CSV to {Path(cfg.output_csv).expanduser()}")

    finally:
        if csv_file:
            csv_file.close()


def main():
    register_third_party_plugins()
    dataset_action_error()


if __name__ == "__main__":
    main()
