from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="smurphcast",
    version="0.0.1",
    description="Boundary‑aware forecasting for percentage KPIs",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Stephen Murphy",
    author_email="<stephenjmurph@gmail.com>",
    url="https://github.com/StephenMurphy/smurphcast",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=["smurphcast", "smurphcast_contrib"],
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.1",
        "scikit-learn>=1.4",
        "lightgbm>=4.3",
        "torch>=2.3",
        "statsmodels>=0.14",
        "matplotlib>=3.8",
        "tqdm>=4.66",
    ],
    entry_points={
        "console_scripts": ["smurphcast = smurphcast.cli:app"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
