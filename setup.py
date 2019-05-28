import sys
import os
import setuptools

version = "0.0.1"


def main():
    setuptools.setup(
        name="project",
        version=version,
        description="description",
        author="Yilang Hu",
        maintainer="Yilang Hu",
        python_requires=">=3.5.*",
        install_requires=[
            "tensorflow-gpu",
            "numpy",
            "pyyaml",
        ]
    )


if __name__ == "__main__":
    main()
