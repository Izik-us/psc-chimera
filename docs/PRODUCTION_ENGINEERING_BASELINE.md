# Production Engineering Baseline

Inspected before implementation on 2026-10-02.

## Source state

- Branch: `chimera-repair`
- HEAD: `5c9ea0ac6b910f35f687def079c1d79235b57c9c` (`Harden pre-production engineering contracts`)
- Worktree: clean

## Existing software contracts

| Area | Observed state |
| --- | --- |
| Package | `setup.py` declares `psc-chimera` version `2.0.0`, Python `>=3.10`, runtime and optional dependencies; `chimera.__version__` separately declares `2.0.0`. |
| Imports | `chimera/__init__.py` eagerly re-exports research/model modules. `scripts/run_design.py` inserts the repository root into `sys.path`. |
| CLI | The only installed console entry point is `chimera-design`; the main design command is `scripts/run_design.py`. |
| Model construction/inference | The CLI constructs `CHIMERAv2.from_pretrained(...)`, then calls `design(...)`; its main inputs are an MSA and source backbone, with demo-only synthetic inputs. |
| Checkpoints/provenance | `chimera/checkpoint.py` defines a versioned `CheckpointManifest`, state-schema and config hashes, plus training/runtime provenance. |
| Configuration | A generic JSON hash helper exists, but no validated, versioned production inference configuration was identified in the package API. |
| Readiness gate | `preproduction_readiness_report(PreProductionGateEvidence(...))` trusts caller-provided booleans and is explicitly documented as an evidence summary, not a self-verifying production gate. |
| Pretrained assets | `scripts/pretrained_manager.py` lists ESM-2, ESMFold, ProteinMPNN, and RFdiffusion download URLs. Shell docs additionally list AlphaFold2 parameters. Existing downloads are accepted by nonzero file size; no asset manifest/hash verification is performed there. |
| Native adapters | OpenFold and RFdiffusion direct adapters explicitly raise `BackendUnavailable` as unwired. CLI adapter boundaries exist. Local EvoFormer and ProteinMPNN-inspired modules are documented as approximations. |
| Local model files | Files were observed under `Models/`, the repository root, and `weights/`; they are not tracked source assets. Their names or presence alone do not establish upstream provenance or adapter compatibility. |
| Dependencies/CI | Dependencies are declared in `setup.py`, `requirements.txt`, and `requirements-dev.txt`. GitHub Actions runs CPU tests on Python 3.10/3.11 and excludes GPU/weight-dependent tests. |
| Tests | Pytest tests reside in `tests/`; the documented/current CI test command is a CPU scientific/integration selection. |
| Documentation | `README.md`, `INSTALL.md`, `docs/PRE_PRODUCTION_ENGINEERING_GATE.md`, and `docs/pretrained_models.md` describe architecture, setup, and scientific boundaries. |

## Production blockers established by inspection

1. A caller can construct passing readiness evidence without executing checks.
2. The script-based design path assumes repository layout and mutates `sys.path`.
3. Downloaded-model identity, upstream checksum status, local hash, and manifest are not verified together by the Python asset manager.
4. The existing adapter documentation correctly does not claim that the local approximations are native upstream models; direct OpenFold and RFdiffusion adapters are not operational.
5. The actual local weight files require individual integrity and compatibility checks before they can be treated as production dependencies.

This report captures observed baseline facts only. It does not assert that any pretrained artifact is authentic, compatible, licensed for a particular use, or biologically validated.
