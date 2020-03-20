"""
Setup script for rnns
"""

from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'torchvision',
    'torch',
    'opencv-python',
    'Pillow',
    'multiprocess',
    'tensorboard'
]

setup(name='rnns',
        version='0.1.0',
        description='A library for image sequence prediction using RNNs',
        author='Steven Lee',
        author_email='stevenl3@andrew.cmu.edu',
        package_dir = {'': '.'},
        packages=['rnns'],
        install_requires=requirements
    )
