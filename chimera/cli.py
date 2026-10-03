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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            _write_json({"software": "psc-chimera", "version": __version__})
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
