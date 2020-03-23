"""
Setup script for prediction_utils
"""

from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'scikit-learn',
    'scikit-image',
    'matplotlib',
    'torchvision',
    'torch',
    'opencv-python',
    'Pillow',
    'multiprocess',
    'pandas',
    'ipdb'
]

setup(name='prediction_utils',
        version='0.1.0',
        description='A library for image sequence prediction using gaussian mixture models',
        author='Steven Lee',
        author_email='stevenl3@andrew.cmu.edu',
        package_dir = {'': '.'},
        packages=['prediction_utils'],
        install_requires=requirements
    )
