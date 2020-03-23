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
import numpy as np

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def Ry(y):
    #returns rotation matrix with a rotation around y axis of y degrees
    return np.array([[math.cos(y), 0, math.sin(y), 0],
                   [0, 1, 0, 0],
                   [-math.sin(y), 0, math.cos(y), 0],
                   [0, 0, 0, 1]])

def Rx(x):
    #returns rotation matrix with a rotation around x axis with x degrees.
    return np.array([[1, 0, 0, 0],
                   [0, math.cos(x), -math.sin(x), 0],
                   [0, math.sin(x), math.cos(x), 0],
                   [0, 0, 0, 1]])

def Rz(z):
    #returns a rotatation matrix with a rotation around the z axis with z degrees
    return np.array([[math.cos(z), -math.sin(z), 0, 0],
                   [math.sin(z), math.cos(z),0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
def Rt(xt,yt,zt):
    #multiplies all rotation matrices for shoulder rotation
    return np.asarray(zt.dot(yt.dot(xt))).reshape(-1)[:12]

def ang(p1,p2,p3):
    #finds angle between 3 points (For the elbow joint)
    p1,p2,p3 = np.array(p1), np.array(p2), np.array(p3)
    ba = p1 - p2
    bc = p3 - p2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    #print(math.radians(angle))
    return angle


def moveArms():
    #getting joint id's for bill
    e, leftShoulder = vrep.simxGetObjectHandle(clientID, 'Bill_leftShoulderJoint', vrep.simx_opmode_oneshot_wait)
    e, rightShoulder = vrep.simxGetObjectHandle(clientID, 'Bill_rightShoulderJoint', vrep.simx_opmode_oneshot_wait)
    e, leftElbow = vrep.simxGetObjectHandle(clientID, 'Bill_leftElbowJoint', vrep.simx_opmode_oneshot_wait)
    e, rightElbow = vrep.simxGetObjectHandle(clientID, 'Bill_rightElbowJoint', vrep.simx_opmode_oneshot_wait)
    e = vrep.simxSetSphericalJointMatrix(clientID, leftShoulder, Rt(Rx(math.pi/2), Ry(-math.pi/2), Rz(0)), vrep.simx_opmode_streaming)


    #reading data
    values = readFile("d.txt")
    #note x and y values between data and vrep are flipped, so what is x in data is y in vrep and vice versa.

    #iterating through all values
    for data in values.split("array"):

        if data == "": continue
        #Left Shoulder
        # Data[18] is shoulder vector, data[19] is elbow vector and data[20] is hand vector
        z = math.atan((data[19][1] - data[18][1])/(data[19][0]-data[18][0]))
        y = math.atan((data[19][2] - data[18][2])/ ((data[19][1] - data[18][1])**2 + (data[19][0] - data[18][0])**2)**0.5)

        e = vrep.simxSetSphericalJointMatrix(clientID, leftShoulder, Rt(Rx(0),Ry(y),Rz(z)), vrep.simx_opmode_streaming)

        #Right Shoulder
        # Data[26] is shoulder vector, data[27] is elbow vector and data[28] is hand vector
        z = math.atan((data[27][1] - data[26][1]) / (data[27][0] - data[26][0]))
        y = math.atan((data[27][2] - data[26][2]) / ((data[27][1] - data[26][1])**2 + (data[27][0] - data[26][0])**2)**0.5)
        #print(math.degrees(z))
        e = vrep.simxSetSphericalJointMatrix(clientID, rightShoulder, Rt(Rx(0),Ry(y),Rz(z)), vrep.simx_opmode_streaming)

        #Left Elbow
        e = vrep.simxSetJointPosition(clientID, leftElbow, -ang(data[18],data[19],data[20]), vrep.simx_opmode_oneshot)

        #Right Elbow
        e = vrep.simxSetJointPosition(clientID, rightElbow, -ang(data[26],data[27],data[28]), vrep.simx_opmode_oneshot)
        time.sleep(1/24)





print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    print ('Connected to remote API server')

    moveArms()

    vrep.simxGetPingTime(clientID)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
