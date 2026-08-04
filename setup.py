from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [l.strip() for l in f
                    if l.strip() and not l.startswith("#")]

setup(
    name="psc-chimera",
    version="2.0.0",
    author="PSC Engineering Pipeline",
    description="CHIMERA: Computational design engine for PSC NRPS engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Izik-us/psc-chimera",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
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
            "chimera-codon=scripts.run_codon_optimizer:main",
        ],
    },
)
