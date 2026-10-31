"""Stable command-line entry point for CHIMERA production utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import ChimeraError
from .model_store import (
    ASSET_ALIASES,
    ensure_asset,
    inspect_asset,
    list_assets,
    verify_asset,
)


def _write_json(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _add_models_parser(subparsers: Any) -> None:
    models = subparsers.add_parser(
        "models", help="list, acquire, inspect, or verify upstream model assets"
    )
    actions = models.add_subparsers(dest="models_action", required=True)
    listing = actions.add_parser("list", help="list declared assets and cache status")
    listing.add_argument("--cache-dir", default=None)
    for action in ("fetch", "verify", "inspect"):
        command = actions.add_parser(
            action, help=f"{action.title()} one declared model asset"
        )
        choices = [*ASSET_ALIASES, *ASSET_ALIASES.values()]
        if action == "verify":
            choices.append("all")
        command.add_argument(
            "asset",
            nargs="?" if action == "verify" else None,
            choices=choices,
            default="all" if action == "verify" else None,
        )
        command.add_argument("--cache-dir", default=None)
        if action == "verify":
            command.add_argument(
                "--offline",
                action="store_true",
                help="verify the local cache only; verification never performs network access",
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chimera", description="PSC-CHIMERA tools")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    version_command = commands.add_parser("version", help="show the software version")
    version_command.set_defaults(command="version")
    _add_models_parser(commands)
    train = commands.add_parser("train", help="train or resume one canonical CHIMERA regime")
    train.add_argument("regime", choices=("representation", "flow", "sequence", "constraint", "objective", "preference"))
    train.add_argument("action", choices=("start", "resume"))
    train.add_argument("--dataset", required=True, help="pre-tensorized CanonicalTrainingBatch JSONL")
    train.add_argument("--checkpoint", required=True, help="single evolving canonical checkpoint path")
    train.add_argument("--architecture", default=None, help="architecture JSON, required for start")
    train.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=1)
    train.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--dataset-manifest", default=None)
    train.add_argument("--dataset-version", default="canonical-training-batch-jsonl-v1")
    train.add_argument("--preprocessing-hash", default=None)
    train.add_argument("--checkpoint-every-steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", default=None)
    checkpoint = commands.add_parser("checkpoint", help="inspect a canonical training checkpoint")
    checkpoint_actions = checkpoint.add_subparsers(dest="checkpoint_action", required=True)
    checkpoint_inspect = checkpoint_actions.add_parser("inspect", help="show active regime and stage history")
    checkpoint_inspect.add_argument("--checkpoint", required=True, type=Path)
    commands.add_parser(
        "production-dependencies",
        help="show the declared production candidate dependency closure",
    )
    gate = commands.add_parser(
        "production-gate", help="execute production checks and emit evidence"
    )
    gate.add_argument("--root", required=True, help="source repository root to inspect")
    gate.add_argument(
        "--output",
        default=None,
        help="new evidence JSON path; existing files are never replaced",
    )
    gate.add_argument("--model-cache", default=None, help="model cache directory")
    design = commands.add_parser("design", help="plan and manage bounded scientific design runs")
    design_actions = design.add_subparsers(dest="design_action", required=True)
    plan = design_actions.add_parser(
        "plan", help="validate a design task and emit a dry-run plan without loading a model"
    )
    plan.add_argument("--task", required=True, help="path to a versioned design-task JSON file")
    plan.add_argument(
        "--output", default=None,
        help="optional new JSON path for the plan; existing files are never replaced",
    )
    prepare = design_actions.add_parser(
        "prepare", help="create an auditable planned/blocked run without generating candidates"
    )
    prepare.add_argument("--task", required=True, help="path to a versioned design-task JSON file")
    prepare.add_argument("--runs-root", default="runs/design", help="directory for durable design runs")
    inspect = design_actions.add_parser("inspect", help="inspect a prepared design run")
    inspect.add_argument("--run-dir", required=True, help="path to a design run directory")
    run = design_actions.add_parser(
        "run", help="prepare and execute one bounded canonical inference task"
    )
    run.add_argument("--task", required=True, help="path to a versioned design-task JSON file")
    run.add_argument("--runs-root", default="runs/design", help="directory for durable design runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            _write_json({"software": "psc-chimera", "version": __version__})
            return 0
        if args.command == "train":
            if args.action == "start" and not args.architecture:
                parser.error("chimera train <regime> start requires --architecture")
            from scripts.train_canonical import main as train_main

            training_args = [
                "--training-jsonl", args.dataset,
                "--checkpoint", str(args.checkpoint),
                "--regime", args.regime,
                "--epochs", str(args.epochs),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--dataset-version", args.dataset_version,
                "--checkpoint-every-steps", str(args.checkpoint_every_steps),
                "--seed", str(args.seed),
            ]
            if args.architecture:
                training_args.extend(["--architecture-json", args.architecture])
            if args.action == "resume":
                training_args.extend(["--resume", str(args.checkpoint)])
            if args.dataset_manifest:
                training_args.extend(["--dataset-manifest", args.dataset_manifest])
            if args.preprocessing_hash:
                training_args.extend(["--preprocessing-hash", args.preprocessing_hash])
            if args.device:
                training_args.extend(["--device", args.device])
            return train_main(training_args)
        if args.command == "checkpoint" and args.checkpoint_action == "inspect":
            import torch

            payload = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or payload.get("artifact_type") != "canonical_training_checkpoint":
                raise ValueError("file is not a canonical training checkpoint")
            training = payload.get("training", {})
            _write_json({
                "artifact_type": payload.get("artifact_type"),
                "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
                "active_regime": training.get("regime"),
                "epoch": training.get("epoch"),
                "global_step": training.get("global_step"),
                "trainable_components": training.get("trainable_components", []),
                "stage_history": training.get("stage_history", []),
                "dataset_version": payload.get("manifest", {}).get("dataset_version"),
                "dataset_hash": payload.get("manifest", {}).get("dataset_hash"),
            })
            return 0
        if args.command == "production-dependencies":
            from .production_gate import production_dependency_inventory

            _write_json(production_dependency_inventory())
            return 0
        if args.command == "models":
            if args.models_action == "list":
                _write_json(list_assets(args.cache_dir))
                return 0
            if args.models_action == "fetch":
                path = ensure_asset(args.asset, args.cache_dir)
                _write_json(
                    {
                        "path": str(path),
                        "verification": verify_asset(args.asset, args.cache_dir),
                    }
                )
                return 0
            if args.models_action == "verify":
                if args.asset == "all":
                    results = [
                        verify_asset(key, args.cache_dir) for key in ASSET_ALIASES
                    ]
                    _write_json({"offline": True, "results": results})
                    return (
                        0
                        if all(
                            result["state"] == "INTEGRITY_VERIFIED"
                            for result in results
                        )
                        else 1
                    )
                result = verify_asset(args.asset, args.cache_dir)
                _write_json(result)
                return 0 if result["state"] == "INTEGRITY_VERIFIED" else 1
            _write_json(inspect_asset(args.asset, args.cache_dir))
            return 0
        if args.command == "design" and args.design_action == "run":
            from .design_runs import run_design_task

            report = run_design_task(
                args.task, runs_root=Path(args.runs_root).expanduser()
            )
            _write_json(report)
            return 0 if report["status"] == "COMPLETED" else 1
        if args.command == "design" and args.design_action == "prepare":
            from .design_runs import create_design_run, load_design_task

            task_path = Path(args.task).expanduser().resolve()
            task = load_design_task(task_path)
            created = create_design_run(
                task, task_file=task_path, runs_root=Path(args.runs_root).expanduser()
            )
            _write_json(created)
            return 0 if created["manifest"]["status"] == "PLANNED" else 1
        if args.command == "design" and args.design_action == "inspect":
            from .design_runs import inspect_design_run

            report = inspect_design_run(args.run_dir)
            _write_json(report)
            return 0
        if args.command == "design" and args.design_action == "plan":
            from .design_runs import build_design_plan, load_design_task

            task_path = Path(args.task).expanduser().resolve()
            task = load_design_task(task_path)
            plan = build_design_plan(task, task_file=task_path)
            if args.output:
                destination = Path(args.output).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with destination.open("x", encoding="utf-8", newline="\n") as stream:
                        json.dump(plan, stream, indent=2, sort_keys=True)
                        stream.write("\n")
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Refusing to replace existing design plan: {destination}"
                    ) from exc
            _write_json(plan)
            return 0 if plan["readiness"] != "BLOCKED" else 1
        if args.command == "production-gate":
            from .production_gate import run_production_gate

            evidence = run_production_gate(
                root=Path(args.root),
                output_path=Path(args.output) if args.output else None,
                model_cache=Path(args.model_cache) if args.model_cache else None,
            )
            _write_json(evidence)
            return 0 if evidence["engineering_status"] == "PASS" else 1
    except ChimeraError as exc:
        _write_json(
            {
                "error": {
                    "category": exc.category,
                    "message": str(exc),
                    "retryable": exc.retryable,
                    "corrective_action": exc.corrective_action,
                }
            }
        )
        return 2
    except OSError as exc:
        _write_json(
            {
                "error": {
                    "category": "operating_system_error",
                    "message": str(exc),
                    "retryable": False,
                }
            }
        )
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
