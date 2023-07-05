from setuptools import setup

setup(
    name="simple-ctc-loss",
    version="0.1",
    packages=['ctc'],
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.0",
        "torch>=2.0"
    ],
    entry_points={
        "console_scripts": [
        ],
    },
    python_requires=">=3.8",
    author="Catalin Negacevschi",
    author_email="negacevschi.catalin@gmail.com",
    description="Simple implementation of the Connectionist Temporal Classification loss",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="",
    url="https://github.com/catalinnega/simple-ctc-loss",
    classifiers=[
    ],
)