# Chef-Bot-Sim

Restuarant simulation in VREP through the Python Remote API

# Setup Instructions:

## Setting up VREP Remote API Environment:

1. Install VREP from: http://www.coppeliarobotics.com/downloads.html

2. Open VREP from terminal by navigating to the installed folder and running the command `./vrep.sh`

3. Open the scene remote Api.ttt from the folder scenes in the git and start the simulation

4. The scene is now setup

## Setting up Python Environment:

1. Create a new project in your desired python IDE.

2. Copy the files vrep.py, vrep.pyc, vrepConst.py, vrepConst.pyc, remoteApi.so, main.py and the models folder from the Restuarant Simulation folder from this git into the desired project folder

3. Change the address variable to point to where the models folder is saved.

# Restuarant Simulation:
 The restuarant simulation can now be run by running the main.py file and it will generate a randomized restuarant environment inside the VREP scene remotely from the python script. 
 
 # Reference:
 Instructions about how the remote api works can be found here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiOverview.htm
 
 The documentation for the python functions in the VREP library can be found here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm
 
 # Features:
 1. Generates Tables, chairs and utensils in randomized positions in the scene
 2. Populates the scene randomly with the required number of people
 3. Generates a robot and teleports it to a random position for testing
 
 # TODO:
 1. Add Behaviors such as waving and nodding (In Progress)
 2. Add facial expressions to visualize the mood of the consumers
 3. Markup the joints of the people with actual pose data from a restuarant
 
