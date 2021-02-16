# Planning

This is a software package for robot planning in a restaurant setting. This package addresses planning in presence of unexpected events. Please use the following instructions to run the code.

- create a virtual machine and activate it (for more info refer to https://docs.python.org/3/library/venv.html)
- install the required packages by using pip3 install -r requirements.txt
- run roscore in a terminal
- go into planning/UnexpectedEvents/scripts folder in two different terminals
- in one terminal run "python observation_input.py" where you can specify what observation the robot gets (observation node). You can also see what action gets selected in this terminal. 
- in the other terminal run "python3 State_Machine.py 90 2 4 True False complex_r robot" where you can run the planner. The planner sends the action that should be executed to the observation node and waits for the observation to be recieved. 
- for more info please refer to the document

# References
[1] A. Mohseni-Kabir, M. Likhachev, and M. Veloso, "Waiting Tables as a Robot Planning Problem," in IJCAI Workshop on AIxFood, 2019.

[2] A. Mohseni-Kabir, M. M. Veloso, and M. Likhachev, “Efficient Robot Planning for Achieving Multiple Independent Partially Observable Tasks that Evolve Over Time,” in ICAPS, 2020

[3] A. Mohseni-Kabir, M. Veloso, and M. Likhachev, "Optimal Planning over Long and Infinite Horizons for Achieving Independent Partially-Observable Tasks that Evolve over Time," submitted to ICAPS, 2021.

[4] A. Mohseni-Kabir, M. Veloso, and M. Likhachev, "Planning in Presence of Unexpected Events," still under development.


