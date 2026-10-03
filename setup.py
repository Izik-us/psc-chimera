from pathlib import Path
import re
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
VERSION_SOURCE = (ROOT / "chimera" / "_version.py").read_text(encoding="utf-8")
VERSION_MATCH = re.search(r'^__version__ = "([^"]+)"$', VERSION_SOURCE, re.MULTILINE)
if VERSION_MATCH is None:
    raise RuntimeError("Unable to read the authoritative version from chimera/_version.py")

# Keep runtime installation lean. Heavy/optional scientific backends and
# developer tooling belong in extras rather than being pulled into every
# environment by pip install .
RUNTIME_REQUIREMENTS = [
    "torch>=2.1.0",
    "einops>=0.7.0",
    "numpy>=1.24.0",
]

setup(
    name="psc-chimera",
    version=VERSION_MATCH.group(1),
    author="PSC Engineering Pipeline",
    description="CHIMERA: Computational design engine for PSC NRPS engineering",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Izik-us/psc-chimera",
    packages=find_packages(),
    package_data={
        "chimera": [
            "model_dependencies.json",
            "architecture_dependencies.json",
        ]
    },
    python_requires=">=3.10",
    install_requires=RUNTIME_REQUIREMENTS,
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0"],
        "data": ["biopython>=1.81"],
        "esm": ["fair-esm>=2.0.0"],
        "retrieval": ["faiss-cpu>=1.7.4"],
        "research": ["scipy>=1.11.0"],
        "viz": ["matplotlib>=3.7.0"],
    },
    entry_points={
        "console_scripts": [
            "chimera=chimera.cli:main",
            "chimera-design=scripts.run_design:main",
        ],
    },
)
