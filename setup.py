from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coscene-converter",
    version="0.1.0",
    author="Zhexuan Yang",
    author_email="zhexuany@gmail.com",
    description="Conversion tools for robotics datasets to MCAP format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhexuany/coscene-tutorials",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow",
        "tensorflow-datasets",
        "foxglove-sdk",
        "numpy",
        "gcsfs",
    ],
    entry_points={
        'console_scripts': [
            'coscene-converter=cli:main',
        ],
    },
)