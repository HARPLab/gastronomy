
import random
try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import math
import random

address = "/home/steelshot/PycharmProjects/Chef-Bot-Sim/models/"

class Table(object):
    def __init__(self,size,position,people = 0):
            #setting variables
            self.size = size
            self.position = position
            self.people = people
            #generating table- table handle stored in self.table
            x, self.table = vrep.simxLoadModel(clientID, address + "diningTable.ttm", 0,
                                  vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.table, -1, (self.position[0], self.position[1], self.position[2] + 0.38), vrep.simx_opmode_oneshot)
            #generating chairs and cutlery
            #chair 1
            x, self.chair1 = vrep.simxLoadModel(clientID,
                                               address + "dining chair.ttm", 0,
                                               vrep.simx_opmode_blocking)

            vrep.simxSetObjectPosition(clientID, self.chair1, -1, (self.position[0], self.position[1] + 0.85, self.position[2] + 0.45),
                                           vrep.simx_opmode_oneshot)
            x, self.plate1 = vrep.simxLoadModel(clientID, address + "plate.ttm", 0,
                                           vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.plate1, -1, (self.position[0], self.position[1] - 0.3, self.position[2] + 0.77), vrep.simx_opmode_oneshot)

            x, self.spoon1 = vrep.simxLoadModel(clientID, address + "spoon.ttm", 0,
                                           vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.spoon1, -1, (self.position[0] + 0.2, self.position[1] - 0.3, self.position[2] + 0.77),
                                       vrep.simx_opmode_oneshot)
            x, self.fork1 = vrep.simxLoadModel(clientID, address + "fork.ttm", 0,
                                          vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.fork1, -1, (self.position[0] - 0.2, self.position[1] - 0.3, self.position[2] + 0.77),
                                       vrep.simx_opmode_oneshot)

            #chair 2
            x, self.chair2 = vrep.simxLoadModel(clientID,
                                               address + "dining chair.ttm", 0,
                                               vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.chair2, -1,
                                           (self.position[0], self.position[1] - 0.85, self.position[2] + 0.45),
                                           vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(clientID, self.chair2, -1, (0, 0, math.pi), vrep.simx_opmode_oneshot)
            x, self.plate2 = vrep.simxLoadModel(clientID, address + "plate.ttm", 0,
                                                vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.plate2, -1,
                                       (self.position[0], self.position[1] + 0.3, self.position[2] + 0.77),
                                       vrep.simx_opmode_oneshot)
            x,self.spoon2 = vrep.simxLoadModel(clientID, address + "spoon.ttm", 0,
                                               vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.spoon2, -1,(self.position[0] - 0.2, self.position[1] + 0.3, self.position[2] + 0.77),
                                       vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(clientID, self.spoon2, -1, (math.pi / 2, 0, math.pi / 2),
                                          vrep.simx_opmode_oneshot)
            x, self.fork2 = vrep.simxLoadModel(clientID, address + "fork.ttm", 0,
                                               vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(clientID, self.fork2, -1,(self.position[0] + 0.2, self.position[1] + 0.3, self.position[2] + 0.77),\
                                       vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(clientID, self.fork2, -1, (math.pi / 2, 0, math.pi / 2),
                                          vrep.simx_opmode_oneshot)
            if self.size == 4:
                #chair 3
                x, self.chair3 = vrep.simxLoadModel(clientID,
                                                    address + "dining chair.ttm", 0,
                                                    vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.chair3, -1,
                                           (self.position[0] + 1, self.position[1], self.position[2] + 0.45),
                                           vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.chair3, -1, (0, 0, -math.pi/2), vrep.simx_opmode_oneshot)

                x, self.plate3 = vrep.simxLoadModel(clientID, address + "plate.ttm", 0,
                                                    vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.plate3, -1,
                                           (self.position[0] + 0.6, self.position[1], self.position[2] + 0.77),
                                           vrep.simx_opmode_oneshot)

                x, self.spoon3 = vrep.simxLoadModel(clientID, address + "spoon.ttm", 0,
                                                    vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.spoon3, -1,
                                           (self.position[0] + 0.6, self.position[1] + 0.2, self.position[2] + 0.77),
                                           vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.spoon3, -1, (math.pi / 2, -math.pi/2, math.pi / 2),
                                              vrep.simx_opmode_oneshot)
                x, self.fork3 = vrep.simxLoadModel(clientID, address + "fork.ttm", 0,
                                                   vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.fork3, -1,
                                           (self.position[0] +0.6, self.position[1] - 0.2, self.position[2] + 0.77), \
                                           vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.fork3, -1, (math.pi / 2, -math.pi/2, math.pi / 2),
                                              vrep.simx_opmode_oneshot)

                #chair 4
                x, self.chair4 = vrep.simxLoadModel(clientID,
                                                address + "dining chair.ttm", 0,
                                                vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.chair4, -1,
                                       (self.position[0] - 1, self.position[1], self.position[2] + 0.45),
                                       vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.chair4, -1, (0, 0, math.pi / 2), vrep.simx_opmode_oneshot)
                x, self.plate4 = vrep.simxLoadModel(clientID, address + "plate.ttm", 0,
                                                vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.plate4, -1,
                                       (self.position[0] - 0.6, self.position[1], self.position[2] + 0.77),
                                       vrep.simx_opmode_oneshot)
                x, self.spoon4 = vrep.simxLoadModel(clientID, address + "spoon.ttm", 0,
                                                    vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.spoon4, -1,
                                           (self.position[0] - 0.6, self.position[1] - 0.2, self.position[2] + 0.77),
                                           vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.spoon4, -1, (math.pi / 2, math.pi / 2, math.pi / 2),
                                              vrep.simx_opmode_oneshot)
                x, self.fork4 = vrep.simxLoadModel(clientID, address + "fork.ttm", 0,
                                                   vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.fork4, -1,
                                           (self.position[0] - 0.6, self.position[1] + 0.2, self.position[2] + 0.77), \
                                           vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.fork4, -1, (math.pi / 2, math.pi / 2, math.pi / 2),
                                              vrep.simx_opmode_oneshot)

            #generating people
            if self.people == 0:
                pass

            elif size == 2 and self.people == 1:
                r = random.randint(0,1)
                if r == 0:
                    x, self.person1 = vrep.simxLoadModel(clientID, address + "Sitting Bill.ttm", 0,
                                                    vrep.simx_opmode_blocking)
                    vrep.simxSetObjectPosition(clientID, self.person1, -1, (self.position[0], self.position[1] + 0.5, self.position[2]),
                                               vrep.simx_opmode_oneshot)
                if r == 1:
                    x, self.person1 = vrep.simxLoadModel(clientID, address + "Sitting Bill.ttm", 0,
                                                    vrep.simx_opmode_blocking)
                    vrep.simxSetObjectPosition(clientID, self.person1, -1, (self.position[0], self.position[1] - 0.5, self.position[2]),
                                               vrep.simx_opmode_oneshot)
                    vrep.simxSetObjectOrientation(clientID, self.person1, -1, (0, 0, math.pi / 2), vrep.simx_opmode_oneshot)
            elif size == 2 and self.people == 2:
                x, self.person2 = vrep.simxLoadModel(clientID, address + "Sitting Bill.ttm", 0,
                                                vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.person2, -1, (self.position[0], self.position[1] - 0.5, self.position[2]), vrep.simx_opmode_oneshot)
                vrep.simxSetObjectOrientation(clientID, self.person2, -1, (0, 0, math.pi / 2), vrep.simx_opmode_oneshot)
                x, self.person1 = vrep.simxLoadModel(clientID, address + "Sitting Bill.ttm", 0,
                                                vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(clientID, self.person1, -1, (self.position[0], self.position[1] + 0.5, self.position[2]), vrep.simx_opmode_oneshot)
            elif size == 4:
                choices = [1,2,3,4]
                for i in range(self.people):
                    r = random.choice(choices)
                    choices.remove(r)
                    if r == 1:
                        x, self.person1 = vrep.simxLoadModel(clientID,address + "Sitting Bill.ttm", 0,
                                                            vrep.simx_opmode_blocking)
                        vrep.simxSetObjectPosition(clientID, self.person1, -1,(self.position[0], self.position[1] + 0.5, self.position[2]),
                                                       vrep.simx_opmode_oneshot)
                    if r == 2:
                        x, self.person2 = vrep.simxLoadModel(clientID,
                                                             address + "Sitting Bill.ttm", 0,
                                                             vrep.simx_opmode_blocking)
                        vrep.simxSetObjectPosition(clientID, self.person2, -1,
                                                   (self.position[0], self.position[1] - 0.5, self.position[2]),
                                                   vrep.simx_opmode_oneshot)
                        vrep.simxSetObjectOrientation(clientID, self.person2, -1, (0, 0, math.pi / 2),
                                                      vrep.simx_opmode_oneshot)
                    if r == 3:
                        x, self.person3 = vrep.simxLoadModel(clientID,
                                                             address + "Sitting Bill.ttm", 0,
                                                             vrep.simx_opmode_blocking)
                        vrep.simxSetObjectPosition(clientID, self.person3, -1,
                                                   (self.position[0] - 0.6, self.position[1], self.position[2]),
                                                   vrep.simx_opmode_oneshot)
                        vrep.simxSetObjectOrientation(clientID, self.person3, -1, (0, 0,0),
                                                      vrep.simx_opmode_oneshot)
                    if r == 4:
                        x, self.person4 = vrep.simxLoadModel(clientID,
                                                             address + "Sitting Bill.ttm", 0,
                                                             vrep.simx_opmode_blocking)
                        vrep.simxSetObjectPosition(clientID, self.person4, -1,
                                                   (self.position[0] + 0.6, self.position[1], self.position[2]),
                                                   vrep.simx_opmode_oneshot)
                        vrep.simxSetObjectOrientation(clientID, self.person4, -1, (0, 0,math.pi),
                                                      vrep.simx_opmode_oneshot)

class Robot(object):
    def __init__(self,position):
        self.position = position
        x, self.robot = vrep.simxLoadModel(clientID,
                                      address + "Sample Chef Bot.ttm", 0,
                                      vrep.simx_opmode_blocking)
        print(2)
        vrep.simxSetObjectPosition(clientID, self.robot, -1,
                                   (self.position[0], self.position[1], self.position[2]),
                                   vrep.simx_opmode_oneshot)
        print(1)


def randomTables():
    d = random.uniform(1.5, 2.5)
    size = random.choice([2,4])
    if size == 2:
        people = random.choice([0,1,2])
    else:
        people = random.choice([0,1,2,3,4])
    t1 = Table(size,[d,d,0],people)

    size = random.choice([2, 4])

    if size == 2:
        people = random.choice([0, 1, 2])
    else:
        people = random.choice([0, 1, 2, 3, 4])
    t2 = Table(size, [-d, d, 0], people)

    size = random.choice([2, 4])
    if size == 2:
        people = random.choice([0, 1, 2])
    else:
        people = random.choice([0, 1, 2, 3, 4])
    t3 = Table(size, [-d, -d, 0], people)

    size = random.choice([2, 4])
    if size == 2:
        people = random.choice([0, 1, 2])
    else:
        people = random.choice([0, 1, 2, 3, 4])
    t4 = Table(size, [d, -d, 0], people)

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    print ('Connected to remote API server')

    randomTables()
    robotX = random.uniform(0,2.5)
    robotY = random.uniform(0,2.5)
    time.sleep(2)
    R1 = Robot([robotX,robotY,0])
    vrep.simxGetPingTime(clientID)

    vrep.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
