#!/usr/bin/env python

from setuptools import setup

# Setup parameters
setup(name = "sarsa_zero",
      version = "0.1.0",
      description = "SARSA Zero using Python and Gym",
      long_description = open("README.md").read(),
      packages = ["sarsa_zero"],
      author = "Dibya Chakravorty",
      author_email = "dibyachakravorty@gmail.com",
      install_requires=[
          "argparse==1.4.0",
          # There's a bug in Box2D environments like LunarLander-v2 or
          # BipedalWalker-v2 (https://github.com/openai/gym/issues/100).
          # Fix it by running the following after this package has been
          # installed:
          # pip uninstall Box2D-kengz
          # git clone https://github.com/pybox2d/pybox2d.git
          # pip install -e pybox2d
          "gym[all]==0.10.5",
          "numpy==1.14.2",
          # Blackjack-v0 needs scipy, but is not in gym[all]'s requirements
          "scipy==1.0.1",
      ],
)
