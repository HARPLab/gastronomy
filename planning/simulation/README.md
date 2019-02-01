# Chef-Bot-Sim

Restuarant simulation in VREP through the Python Remote API

# Setup Instructions:

## Setting up VREP Remote API Environment:

1. Install VREP from: http://www.coppeliarobotics.com/downloads.html

2. Open VREP from terminal by navigating to the installed folder and running the command `./vrep.sh`

3. Create a new scene

4. Add a small cuboid to the scene by right clicking anywhere in the scene and then add -> primitive shape -> cuboid. Create the cuboid with the default settings

5. Right click on the cuboid on the scene and add a threaded child script by add -> associated child script -> threaded

6. Open the created script by double clicking the small page icon next to the cuboid in the scene hierachy

7. Add the line `simRemoteApi.start(19999)` inside the function sysCall_threadmain

8. The client side is now setup

## Setting up Python Environment:

1. Create a new project in your desired python IDE.

2. Copy the files vrep.py, vrep.pyc, vrepConst.py, vrepConst.pyc, remoteApi.so and main.py from the Restuarant Simulation form this git into the project folder

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
 
