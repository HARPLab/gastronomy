# Planning

This is a software package for robot planning in a restaurant setting. This package addresses planning in presence of unexpected events. Please use the following instructions to run the code.

- create a virtual machine and activate it (for more info refer to https://docs.python.org/3/library/venv.html)
- install the required packages by using pip3 install -r requirements.txt
- run roscore in a terminal
- go into planning/scripts folder in two different terminals
- in one terminal run "python observation_input.py" where you can specify what observation the robot gets
- in the other terminal run "python3 State_Machine.py 90 3 4 True False complex_no_op_hybrid_avg robot" where you can run the planner
- for more info please refer to the document 
