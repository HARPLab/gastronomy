# Planning in grid-world

This is a software package for running the task-switching code on a grid-world restaurant simulation.

## Requirements

Python3

## Installation

1. Change directory to "planning": `cd planning`
2. Install virtualenv for python3: `python3 -m pip install virtualenv`
3. Create a virtualenv: `python3 -m virtualenv env`
4. Activate the virtualenv: `source env/bin/activate`
5. Install numpy, gym, matplotlib and ipdb packages through pip install: `pip install -r requirements.txt`
6. Run `python3 State_Machine.py 90 5 4 True False complex_no_op_hybrid_avg belief_mix`  
