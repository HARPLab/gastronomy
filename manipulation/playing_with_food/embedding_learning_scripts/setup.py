"""
Setup script for food_embeddings in playing_with_food repo
"""

from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'torchvision',
    'torch',
    'opencv-python',
    'tensorboard'
]

setup(name='food_embeddings',
        version='0.1.0',
        description='A library for making embeddings of food items',
        package_dir = {'': '.'},
        packages=['food_embeddings'],
        install_requires=requirements
    )
