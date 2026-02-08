"""
GuardRAG: Guarded Retrieval-Augmented Generation
Setup configuration for the project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="guardrag",
    version="1.0.0",
    author="GuardRAG Team",
    description="Guarded Retrieval-Augmented Generation with DPO and Defense Mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mg-2321/GuardRAG",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "guardrag=main:main",
        ],
    },
)
