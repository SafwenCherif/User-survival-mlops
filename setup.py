from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Titanic-Survival-Prediction",
    version="0.1",
    author="Safwen",
    packages=find_packages(),
    install_requires = requirements,
)