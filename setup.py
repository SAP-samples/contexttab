from collections import defaultdict
import pathlib
import subprocess
from typing import List

from setuptools import find_packages, setup

EXTRAS_SEPARATOR = "#"


# Function to read any lines including version and requirements from the .txt files
def get_version(file_path: pathlib.Path) -> str:
    with open(file_path, "r") as f:
        return f.read().strip()


# Get requirements from requirements.txt file
def get_requirements(file_path: pathlib.Path) -> List[str]:
    with open(file_path, "r") as f:
        return [
            line.strip()
            for line in f.readlines()
            if len(line.strip()) > 0
            and not (line.startswith("#") or line.startswith("-r"))
            and not EXTRAS_SEPARATOR in line
        ]


def get_git_commit_id() -> str:
    try:
        return subprocess.check_output(["git", "describe", "--always"]).decode("utf-8").strip()
    except Exception:
        return "Unknown"


HERE = pathlib.Path(__file__).parent
requirements_file = HERE / "requirements.txt"
version_file = HERE / "version.txt"

version = get_version(version_file)
requirements = get_requirements(requirements_file)

setup(
    name="contexttab",
    version=version,
    author="contexttab",
    author_email="",
    description="ConTextTab deep learning model",
    packages=find_packages(include=["contexttab", "contexttab.*"]),
    package_data={
        "contexttab": [
            "checkpoints/l2/base.pt", "*.lfs"
        ],
    },
    include_package_data=True,
    # test with: python setup.py bdist_wheel
    classifiers=["Git :: Commit ID :: {}".format(get_git_commit_id()), "Programming Language :: Python :: 3.11"],
    python_requires=">=3.11",
    zip_safe=False,
    url="",
    install_requires=requirements,
)