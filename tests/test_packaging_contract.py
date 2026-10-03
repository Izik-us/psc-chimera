import json
import subprocess
import sys
import venv
from pathlib import Path

from chimera import __version__
from chimera.checkpoint import CheckpointManifest
from chimera.cli import main
from chimera.model_store import ASSET_ALIASES


def test_package_and_checkpoint_share_authoritative_version():
    assert CheckpointManifest().chimera_version == __version__


def test_cli_module_help_is_available(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help must exit successfully")
    assert "production-gate" in capsys.readouterr().out


def test_isolated_import_does_not_depend_on_repository_cwd(tmp_path: Path):
    command = [
        sys.executable,
        "-I",
        "-c",
        "import chimera; print(chimera.__version__)",
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(
            "Package is not importable outside the repository after installation: "
            + completed.stderr.strip()
        )
    assert completed.stdout.strip() == __version__


def test_regular_install_exposes_import_and_cli_outside_source_tree(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    environment = tmp_path / "install-env"
    venv.EnvBuilder(system_site_packages=True).create(environment)
    if sys.platform == "win32":
        python = environment / "Scripts" / "python.exe"
        executable = environment / "Scripts" / "chimera.exe"
    else:
        python = environment / "bin" / "python"
        executable = environment / "bin" / "chimera"

    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-build-isolation",
            str(repository),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert installed.returncode == 0, installed.stderr

    imported = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata, chimera; "
            "assert chimera.__version__ == importlib.metadata.version('psc-chimera'); "
            "print(chimera.__version__)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == __version__

    version = subprocess.run(
        [str(executable), "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f"chimera {__version__}"

    model_list = subprocess.run(
        [
            str(executable),
            "models",
            "list",
            "--cache-dir",
            str(tmp_path / "empty-cache"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert model_list.returncode == 0, model_list.stderr
    assets = json.loads(model_list.stdout)
    assert len(assets) == len(ASSET_ALIASES)
