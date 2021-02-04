# Planning

This is a software package for robot planning in a restaurant setting. The details of the planner and the restaurant model is in [1,2,3,4].

- create a virtual machine and activate it (for more info refer to https://docs.python.org/3/library/venv.html)
- install the required packages by using pip3 install -r requirements.txt
- go into planning/scripts folder in two different terminals
- run roscore
- in one terminal run "python observation_input.py" where you can specify what observation the robot gets
- in the other terminal run "python3 State_Machine.py 90 3 4 True False complex_no_op_hybrid_avg robot" where you can run the planner
- for more info please refer to the document

# References
[1] A. Mohseni-Kabir, M. Likhachev, and M. Veloso, "Waiting Tables as a Robot Planning Problem," in IJCAI Workshop on AIxFood, 2019.

[2] A. Mohseni-Kabir, M. M. Veloso, and M. Likhachev, “Efficient Robot Planning for Achieving Multiple Independent Partially Observable Tasks that Evolve Over Time,” in ICAPS, 2020

[3] A. Mohseni-Kabir, M. Veloso, and M. Likhachev, "Optimal Planning over Long and Infinite Horizons for Achieving Independent Partially-Observable Tasks that Evolve over Time," submitted to ICAPS, 2021.

[4] A. Mohseni-Kabir, M. Veloso, and M. Likhachev, "Planning in Presence of Unexpected Events," still under development.


