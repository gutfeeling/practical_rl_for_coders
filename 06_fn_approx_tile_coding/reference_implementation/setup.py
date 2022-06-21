#!/usr/bin/env python

from setuptools import setup

# Setup parameters
setup(name = "fn_approx_tile_coding",
      version = "0.1.0",
      description = "RL agent using function approximation with Tile Coding",
      long_description = open("README.md").read(),
      packages = ["tile_coding", "fn_approx_tile_coding"],
      author = "Dibya Chakravorty, Richard S. Sutton",
      author_email = "dibyachakravorty@gmail.com, rsutton@ualberta.ca",
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
          "numpy==1.22.0",
          # Blackjack-v0 needs scipy, but is not in gym[all]'s requirements
          "scipy==1.0.1",
      ],
)
