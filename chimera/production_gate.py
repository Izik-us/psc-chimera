"""Self-executing production evidence collection for PSC-CHIMERA."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import venv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._version import __version__
from .model_store import (
    ASSET_ALIASES,
    DEPENDENCY_MANIFEST,
    list_assets,
    verify_asset,
)


GATE_VERSION = "1"
_TEST_COUNT = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|errors?|skipped)")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    required: bool = True
    evidence: dict[str, Any] | None = None


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_check(root: Path) -> tuple[Check, dict[str, Any]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        identity = {"commit": None, "branch": None, "worktree_clean": None}
        return Check("source_identity", "UNAVAILABLE", str(exc), evidence=identity), identity
    identity = {
        "commit": commit or None,
        "branch": branch or None,
        "worktree_clean": not bool(status.strip()),
        "worktree_status": status.splitlines(),
    }
    clean = identity["worktree_clean"]
    result = Check(
        "source_identity",
        "PASS" if commit and clean else "FAIL",
        "Git commit and clean worktree recorded"
        if commit and clean
        else "worktree is dirty; evidence identifies the commit but is not release-ready",
        evidence=identity,
    )
    return result, identity


def _package_check(root: Path) -> Check:
    try:
        with tempfile.TemporaryDirectory(prefix="chimera-package-gate-") as temporary:
            temporary_root = Path(temporary)
            environment = temporary_root / "environment"
            venv.EnvBuilder(system_site_packages=True).create(environment)
            if os.name == "nt":
                environment_python = environment / "Scripts" / "python.exe"
                chimera_command = environment / "Scripts" / "chimera.exe"
            else:
                environment_python = environment / "bin" / "python"
                chimera_command = environment / "bin" / "chimera"
            install_command = [
                str(environment_python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                str(root),
            ]
            installed = subprocess.run(
                install_command,
                cwd=temporary_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
            if installed.returncode != 0:
                return Check(
                    "package_import",
                    "FAIL",
                    installed.stderr.strip() or "regular package installation failed",
                    evidence={
                        "install_command": install_command,
                        "returncode": installed.returncode,
                        "stdout_tail": installed.stdout[-2000:],
                    },
                )
            code = (
                "import importlib.metadata, json, chimera; "
                "print(json.dumps({'version': chimera.__version__, "
                "'distribution_version': importlib.metadata.version('psc-chimera'), "
                "'module': chimera.__file__}))"
            )
            imported = subprocess.run(
                [str(environment_python), "-I", "-c", code],
                cwd=temporary_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=45,
            )
            if imported.returncode != 0:
                return Check(
                    "package_import",
                    "FAIL",
                    imported.stderr.strip() or "isolated installed-package import failed",
                    evidence={
                        "install_command": install_command,
                        "returncode": imported.returncode,
                    },
                )
            try:
                identity = json.loads(imported.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                return Check(
                    "package_import",
                    "FAIL",
                    f"invalid isolated import output: {exc}",
                    evidence={"stdout": imported.stdout},
                )
            cli = subprocess.run(
                [str(chimera_command), "--version"],
                cwd=temporary_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=45,
            )
            if cli.returncode != 0 or cli.stdout.strip() != f"chimera {__version__}":
                return Check(
                    "package_import",
                    "FAIL",
                    cli.stderr.strip() or "installed chimera console entry point failed",
                    evidence={
                        "install_command": install_command,
                        "cli_returncode": cli.returncode,
                        "cli_stdout": cli.stdout.strip(),
                    },
                )
            model_cli = subprocess.run(
                [
                    str(chimera_command),
                    "models",
                    "list",
                    "--cache-dir",
                    str(temporary_root / "empty-model-cache"),
                ],
                cwd=temporary_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=45,
            )
            try:
                model_inventory = json.loads(model_cli.stdout)
            except json.JSONDecodeError:
                model_inventory = None
            if (
                model_cli.returncode != 0
                or not isinstance(model_inventory, list)
                or len(model_inventory) != len(ASSET_ALIASES)
            ):
                return Check(
                    "package_import",
                    "FAIL",
                    model_cli.stderr.strip()
                    or "installed model inventory/resource smoke failed",
                    evidence={
                        "cli_returncode": model_cli.returncode,
                        "cli_stdout_tail": model_cli.stdout[-2000:],
                    },
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("package_import", "UNAVAILABLE", str(exc))

    module_path = identity.get("module")
    installed_tree = False
    if isinstance(module_path, str):
        try:
            Path(module_path).resolve().relative_to(environment.resolve())
            installed_tree = True
        except ValueError:
            pass
    matches = (
        identity.get("version") == __version__
        and identity.get("distribution_version") == __version__
        and installed_tree
    )
    return Check(
        "package_import",
        "PASS" if matches else "FAIL",
        "regular no-dependency installation, isolated import, and console entry point verified"
        if matches
        else "installed package and distribution metadata identity do not match",
        evidence={
            "version": identity.get("version"),
            "distribution_version": identity.get("distribution_version"),
            "module": module_path,
            "module_from_install_target": installed_tree,
            "packaged_model_count": len(model_inventory),
        },
    )


def _compile_check(root: Path) -> Check:
    command = [
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "chimera",
        "data",
        "scripts",
    ]
    try:
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False, timeout=180
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("compile", "UNAVAILABLE", str(exc), evidence={"command": command})
    return Check(
        "compile",
        "PASS" if completed.returncode == 0 else "FAIL",
        "compileall completed" if completed.returncode == 0 else completed.stderr.strip(),
        evidence={"command": command, "returncode": completed.returncode},
    )


def _test_check(root: Path) -> Check:
    command = [sys.executable, "-m", "pytest", "-q"]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
    except FileNotFoundError as exc:
        return Check(
            "tests",
            "UNAVAILABLE",
            f"test runner unavailable: {exc}",
            evidence={"command": command},
        )
    except subprocess.TimeoutExpired as exc:
        return Check(
            "tests",
            "FAIL",
            "test suite exceeded the 30-minute gate timeout",
            evidence={
                "command": command,
                "duration_seconds": time.monotonic() - started,
                "partial_output": str(exc.stdout or "")[-4000:],
            },
        )
    except OSError as exc:
        return Check("tests", "UNAVAILABLE", str(exc), evidence={"command": command})

    output = f"{completed.stdout}\n{completed.stderr}"
    summary_lines = [
        line.strip()
        for line in output.splitlines()
        if " passed" in line or " failed" in line or " skipped" in line or " error" in line
    ]
    summary = summary_lines[-1] if summary_lines else ""
    values = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    matches = list(_TEST_COUNT.finditer(summary))
    for match in matches:
        kind = match.group("kind")
        key = "errors" if kind.startswith("error") else kind
        values[key] += int(match.group("count"))
    parsed = bool(matches and any(values.values()))
    tests_passed = (
        completed.returncode == 0
        and parsed
        and values["passed"] > 0
        and values["failed"] == 0
        and values["errors"] == 0
        and values["skipped"] == 0
    )
    status = "PASS" if tests_passed else "FAIL"
    if not parsed:
        detail = "pytest did not emit a parseable non-empty test summary"
    elif values["skipped"]:
        detail = "tests were skipped; skips are not counted as passes"
    elif completed.returncode:
        detail = "pytest returned a non-zero exit status"
    else:
        detail = "full pytest suite completed without failures or skips"
    return Check(
        "tests",
        status,
        detail,
        evidence={
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "test_count": sum(values.values()),
            "passed": values["passed"],
            "failures": values["failed"] + values["errors"],
            "skipped": values["skipped"],
            "summary": summary,
            "output_tail": output[-4000:],
        },
    )


def _manifest_check(model_cache: Path | None) -> tuple[Check, list[dict[str, Any]], str | None]:
    manifest_path = Path(__file__).with_name("model_dependencies.json")
    try:
        manifest_bytes = manifest_path.read_bytes()
        document = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return Check("external_dependency_manifest", "FAIL", str(exc)), [], None
    if (
        document.get("schema_version") != 1
        or document.get("manifest_id") != DEPENDENCY_MANIFEST.get("manifest_id")
        or not isinstance(document.get("dependencies"), list)
    ):
        return Check(
            "external_dependency_manifest",
            "FAIL",
            "dependency manifest schema or package identity mismatch",
        ), [], hashlib.sha256(manifest_bytes).hexdigest()
    dependencies = document["dependencies"]
    required_fields = (
        "artifact_id",
        "model_family",
        "model_variant",
        "upstream_project",
        "upstream_repository",
        "upstream_revision",
        "filename",
        "source_url",
        "format",
        "architecture",
        "license",
        "license_url",
        "required",
    )
    ids = [entry.get("artifact_id") for entry in dependencies if isinstance(entry, dict)]
    if (
        len(ids) != len(dependencies)
        or len(ids) != len(set(ids))
        or any(
            any(not entry.get(field) for field in required_fields[:-1])
            or not isinstance(entry.get("required"), bool)
            for entry in dependencies
            if isinstance(entry, dict)
        )
    ):
        return (
            Check(
                "external_dependency_manifest",
                "FAIL",
                "dependency entries are missing identities, source/license data, or unique IDs",
            ),
            [],
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
    cache_entries = list_assets(model_cache)
    rows = []
    blockers = []
    known_by_id = {
        artifact_id: key for key, artifact_id in ASSET_ALIASES.items()
    }
    for asset in cache_entries:
        metadata = asset["metadata"]
        verification = (
            verify_asset(known_by_id[asset["artifact_id"]], model_cache)
            if asset["present"] or asset["manifest_present"]
            else asset["verification"]
        )
        row = {
            "artifact_id": asset["artifact_id"],
            "required": metadata["required"],
            "state": verification["state"],
            "integrity": verification["integrity"],
            "compatibility": verification["compatibility"],
            "sha256": verification.get("sha256"),
            "manifest_sha256": verification.get("manifest_sha256"),
        }
        rows.append(row)
        if metadata["required"] and verification["state"] != "INTEGRITY_VERIFIED":
            blockers.append(asset["artifact_id"])
        if metadata["required"] and verification.get("compatibility") != "VALID":
            blockers.append(f"{asset['artifact_id']} adapter compatibility")
    workspace_names = {
        "esm2_t30_150M_UR50D.pt",
        "esm2_t30_150M_UR50D-contact-regression.pt",
        "esmfold_3B_v1.pt",
        "v_48_020.pt",
        "proteinmpnn_v48_020.pt",
        "Base_ckpt.pt",
        "rfdiffusion_base.pt",
        "alphafold_params_2022-12-06.tar",
        "poet_weights.pt",
    }
    unregistered_files = []
    for relative in ("", "Models", "weights"):
        directory = Path(__file__).resolve().parent.parent / relative
        for filename in sorted(workspace_names):
            candidate = directory / filename
            if not candidate.is_file():
                continue
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            unregistered_files.append(
                {
                    "path": str(candidate),
                    "size_bytes": candidate.stat().st_size,
                    "local_sha256": digest.hexdigest(),
                    "status": "UNREGISTERED_LOCAL_FILE",
                    "upstream_origin": "UNVERIFIED",
                }
            )
    return (
        Check(
            "external_dependency_manifest",
            "PASS" if not blockers else "FAIL",
            "dependency manifest parsed; no unavailable required pretrained assets"
            if not blockers
            else "required model dependency is missing, corrupt, or compatibility-unknown",
            evidence={
                "manifest_id": document["manifest_id"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "dependencies": rows,
                "unregistered_workspace_files": unregistered_files,
                "blockers": blockers,
            },
        ),
        rows,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _unavailable(name: str, explanation: str) -> Check:
    return Check(name, "UNAVAILABLE", explanation)


def _runtime_checks() -> list[Check]:
    try:
        import torch
    except ImportError as exc:
        return [
            Check("cpu_runtime", "FAIL", f"PyTorch runtime unavailable: {exc}"),
            Check(
                "cuda_runtime",
                "UNAVAILABLE",
                "CUDA runtime could not be evaluated because PyTorch is unavailable",
                required=False,
            ),
        ]
    try:
        output = torch.tensor([1.0, 2.0], device="cpu").sum()
        if not torch.isfinite(output):
            raise RuntimeError("CPU tensor smoke produced a non-finite result")
    except RuntimeError as exc:
        return [
            Check("cpu_runtime", "FAIL", f"CPU tensor smoke failed: {exc}"),
            Check(
                "cuda_runtime",
                "UNAVAILABLE",
                "CUDA runtime was not evaluated after CPU smoke failure",
                required=False,
            ),
        ]
    if not torch.cuda.is_available():
        cuda = Check(
            "cuda_runtime",
            "UNAVAILABLE",
            "CUDA hardware/runtime is not available in this environment",
            required=False,
            evidence={"cuda_available": False},
        )
    else:
        try:
            output = torch.tensor([1.0, 2.0], device="cuda").sum()
            torch.cuda.synchronize()
            if not torch.isfinite(output):
                raise RuntimeError("CUDA tensor smoke produced a non-finite result")
        except RuntimeError as exc:
            cuda = Check(
                "cuda_runtime",
                "FAIL",
                f"CUDA tensor smoke failed: {exc}",
                required=True,
                evidence={"cuda_available": True},
            )
        else:
            cuda = Check(
                "cuda_runtime",
                "PASS",
                "CUDA tensor smoke completed",
                required=True,
                evidence={"cuda_available": True, "device": torch.cuda.get_device_name(0)},
            )
    return [
        Check(
            "cpu_runtime",
            "PASS",
            "CPU tensor smoke completed",
            evidence={"torch_version": str(torch.__version__), "device": "cpu"},
        ),
        cuda,
    ]


def _migration_policy_check() -> Check:
    from .checkpoint import CheckpointManifest

    supported = []
    for version in (1, 2, 3, 4):
        CheckpointManifest(format_version=version).validate()
        supported.append(version)
    return Check(
        "migration_policy",
        "PASS",
        "known checkpoint formats validate; migration remains explicit",
        evidence={
            "supported_formats": supported,
            "explicit_migration_ids": ["v4-serialization-metadata-v0-to-v1"],
            "selected_checkpoint": None,
        },
    )


def _evidence_identity(evidence: dict[str, Any]) -> str:
    payload = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    return _canonical_hash(payload)


def run_production_gate(
    root: os.PathLike[str] | str | None = None,
    *,
    output_path: os.PathLike[str] | str | None = None,
    model_cache: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Execute production checks and write a content-addressed evidence record.

    No check result is supplied by the caller. Unavailable artifact-specific
    checks block release readiness rather than being converted to success.
    """
    repository_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parent.parent
    )
    checks: list[Check] = []
    source_check, source = _git_check(repository_root)
    checks.append(source_check)
    checks.extend(
        [
            _package_check(repository_root),
            _compile_check(repository_root),
            _test_check(repository_root),
        ]
    )
    dependency_check, dependency_rows, dependency_manifest_sha256 = _manifest_check(
        Path(model_cache).expanduser().resolve() if model_cache is not None else None,
    )
    checks.append(dependency_check)
    checks.extend(_runtime_checks())
    checks.append(_migration_policy_check())
    checks.extend(
        [
            _unavailable(
                "configuration",
                "no selected, versioned production inference configuration was found",
            ),
            _unavailable(
                "checkpoint_compatibility",
                "no selected CHIMERA checkpoint and compatibility manifest were supplied by the repository",
            ),
            _unavailable(
                "artifact_integrity",
                "no production model artifact was selected for manifest/hash verification",
            ),
            _unavailable(
                "provenance",
                "no production model artifact exists to link source, configuration, dataset, and upstream dependencies",
            ),
            _unavailable(
                "inference_smoke",
                "no validated inference artifact/configuration is available for an honest production smoke",
            ),
            _unavailable(
                "determinism_smoke",
                "no validated inference artifact/configuration is available for a reproducibility comparison",
            ),
        ]
    )

    failures = [
        check.name
        for check in checks
        if check.required and check.status != "PASS"
    ]
    test_result = next((check for check in checks if check.name == "tests"), None)
    test_evidence = test_result.evidence if test_result and test_result.evidence else {}
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "software_version": __version__,
    }
    try:
        distribution_version = importlib.metadata.version("psc-chimera")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    environment["distribution_version"] = distribution_version
    environment["declared_runtime_versions"] = {}
    for distribution in (
        "torch",
        "numpy",
        "transformers",
        "fair-esm",
        "einops",
        "biopython",
        "faiss-cpu",
    ):
        try:
            environment["declared_runtime_versions"][distribution] = (
                importlib.metadata.version(distribution)
            )
        except importlib.metadata.PackageNotFoundError:
            environment["declared_runtime_versions"][distribution] = None
    generated_at = datetime.now(timezone.utc).isoformat()
    timestamp_for_filename = (
        generated_at.replace("-", "").replace(":", "").replace(".", "")
    )
    if output_path is None:
        evidence_dir = repository_root / "production-evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        destination = evidence_dir / f"production-gate-{timestamp_for_filename}.json"
    else:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "gate_version": GATE_VERSION,
        "software_version": __version__,
        "git_commit": source.get("commit"),
        "worktree_status": source.get("worktree_status"),
        "worktree_clean": source.get("worktree_clean"),
        "artifact_identity": None,
        "checkpoint_identity": None,
        "configuration_identity": None,
        "environment_identity": _canonical_hash(environment),
        "test_command": test_evidence.get("command"),
        "test_result": test_result.status if test_result else "UNAVAILABLE",
        "test_count": test_evidence.get("test_count", 0),
        "failure_count": test_evidence.get("failures"),
        "skip_count": test_evidence.get("skipped"),
        "compile_result": next(
            (check.status for check in checks if check.name == "compile"), "UNAVAILABLE"
        ),
        "import_result": next(
            (check.status for check in checks if check.name == "package_import"), "UNAVAILABLE"
        ),
        "inference_smoke_result": "UNAVAILABLE",
        "determinism_result": "UNAVAILABLE",
        "artifact_integrity_result": "UNAVAILABLE",
        "checkpoint_compatibility_result": "UNAVAILABLE",
        "dependency_result": dependency_check.status,
        "dependency_manifest_sha256": dependency_manifest_sha256,
        "dependencies": dependency_rows,
        "timestamp": generated_at,
        "runtime_metadata": environment,
        "checks": [asdict(check) for check in checks],
        "engineering_status": "PASS" if not failures else "BLOCKED",
        "blockers": failures,
        "scientific_status": "NOT SCIENTIFICALLY VALIDATED",
        "biological_validation": "NOT ESTABLISHED",
        "evidence_path": str(destination),
    }
    evidence["evidence_sha256"] = _evidence_identity(evidence)

    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to replace immutable production evidence: {destination}"
        ) from exc
    return evidence
