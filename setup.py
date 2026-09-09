from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

# Keep runtime installation lean. Heavy/optional scientific backends and
# developer tooling belong in extras rather than being pulled into every
# environment by pip install .
RUNTIME_REQUIREMENTS = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "fair-esm>=2.0.0",
    "einops>=0.7.0",
    "biopython>=1.81",
    "faiss-cpu>=1.7.4",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "pandas>=2.0.0",
    "h5py>=3.9.0",
    "matplotlib>=3.7.0",
    "tqdm>=4.65.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
]

setup(
    name="psc-chimera",
    version="2.0.0",
    author="PSC Engineering Pipeline",
    description="CHIMERA: Computational design engine for PSC NRPS engineering",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Izik-us/psc-chimera",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=RUNTIME_REQUIREMENTS,
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0"],
        "viz": ["seaborn>=0.12.0"],
        "md": ["openmm>=8.0.0", "mdtraj>=1.9.9"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "chimera-design=scripts.run_design:main",
        ],
    },
)
