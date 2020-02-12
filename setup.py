from setuptools import find_packages, setup

setup(
    name="causal-metarl",
    version=0.1,
    description="A reimplementation of 'Causal Reasoning from Meta-reinforcement learning'",
    author="Neel Kant",
    author_email="kantneel@berkeley.edu",
    python_requires=">=3.6.3",
    packages=find_packages("src"),
)
