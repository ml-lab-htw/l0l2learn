from pathlib import Path
from setuptools import setup, find_packages


setup(
    name="l0l2learn",
    version="0.1.1",
    description="Cardinality- and budget-constrained feature selection for logistic regression using mixed-integer conic optimization",
    author="Ricardo Knauer",
    author_email="ricardo.knauer@htw-berlin.de",
    url="https://github.com/ml-lab-htw/l0l2learn",
    packages=find_packages(),
    install_requires=[
        "joblib",
        "mosek",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm"
    ],
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)