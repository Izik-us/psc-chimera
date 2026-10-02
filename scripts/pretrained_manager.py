"""Compatibility CLI for the identity-addressed CHIMERA model store.

New installations should prefer ``chimera models``.  This script remains for
users of the earlier ``--asset`` interface, but never downloads assets unless
an asset is named explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chimera.model_store import (
    PUBLIC_ASSETS,
    cache_root,
    ensure_asset,
    inspect_asset,
    list_assets,
    load_esm2,
    verify_asset,
)


def ensure_public_assets(root=None) -> dict[str, Path]:
    return {key: ensure_asset(key, root) for key in PUBLIC_ASSETS}


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PSC-CHIMERA upstream model asset manager")
    parser.add_argument("--root", default=None, help="model cache root (or CHIMERA_MODEL_CACHE)")
    parser.add_argument("--asset", choices=[*PUBLIC_ASSETS, "all"], default=None)
    parser.add_argument("--fetch", action="store_true", help="explicitly acquire the named asset")
    parser.add_argument("--verify", action="store_true", help="verify cached integrity without network")
    parser.add_argument("--inspect", action="store_true", help="inspect the declared asset and local state")
    parser.add_argument("--offline", action="store_true", help="verify local cached assets only")
    parser.add_argument("--load-esm", action="store_true", help="load an already-cached, verified ESM-2 checkpoint")
    parser.add_argument("--force", action="store_true", help="deprecated; immutable asset identities cannot be overwritten")
    args = parser.parse_args(argv)
    if args.force:
        parser.error("--force is not supported: immutable artifact identities cannot be overwritten")
    if sum((args.fetch, args.verify, args.inspect, args.load_esm)) > 1:
        parser.error("choose only one of --fetch, --verify, --inspect, or --load-esm")
    if args.load_esm:
        model, _, path = load_esm2(args.root)
        _print({"status": "LOADED", "artifact_id": PUBLIC_ASSETS["esm2_t30_150m"].artifact_id, "path": str(path), "device": str(next(model.parameters()).device)})
        return 0
    if args.offline and not args.verify:
        parser.error("--offline may only be used with --verify")
    if args.verify and (args.asset is None or args.asset == "all"):
        results = [verify_asset(key, args.root) for key in PUBLIC_ASSETS]
        _print({"offline": True, "results": results})
        return 0 if all(result["state"] == "INTEGRITY_VERIFIED" for result in results) else 1
    if args.asset == "all" and args.fetch:
        results = []
        for key in PUBLIC_ASSETS:
            path = ensure_asset(key, args.root)
            results.append({"asset": key, "path": str(path)})
        _print(results)
        return 0
    if args.asset is None or args.asset == "all":
        _print(list_assets(args.root))
        return 0
    if args.fetch:
        _print({"path": str(ensure_asset(args.asset, args.root)), "verification": verify_asset(args.asset, args.root)})
        return 0
    if args.verify:
        result = verify_asset(args.asset, args.root)
        _print(result)
        return 0 if result["state"] == "INTEGRITY_VERIFIED" else 1
    if args.inspect:
        _print(inspect_asset(args.asset, args.root))
        return 0
    # Preserve the old explicit ``--asset NAME`` acquisition behavior.
    if args.asset is not None:
        _print({"path": str(ensure_asset(args.asset, args.root)), "verification": verify_asset(args.asset, args.root)})
        return 0
    _print({"cache_root": str(cache_root(args.root)), "assets": list_assets(args.root)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
