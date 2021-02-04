import numpy as np
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=['cucumber'])
    parser.add_argument('--thicknesses', nargs='+', type=float, default=[0.01],
                        help='Thickness for cut slices')
    parser.add_argument('--dmp_weights_file_path', type=str, default='')
    args = parser.parse_args()

    print(os.path.dirname(os.path.realpath(__file__)))

    print('Starting robot')
    fa=FrankaArm()
    fa2=FrankaArm(robot_num=2, init_node=False)   

    if not args.dmp_weights_file_path:
        args.dmp_weights_file_path='.pkl'
        
    file = open(args.dmp_weights_file_path,"rb")
    pose_dmp_info = pickle.load(file)

   
    import pdb; pdb.set_trace()  
    '''
    Order of cuts:
    -cut 1: straight back and forth, thickness = 3mm 
    -cut 2: straight back and forth, thickness = 5mm 
    -cut 3: straight back and forth, thickness = 10mm 
    -cut 4: straight back and forth, thickness = 20mm 
    -cut 5: straight back and forth, thickness = 50mm 
    -cut 6: 30 deg L angled slice, thickness = 30mm 
    -cut 7: 30 deg L angled slice, thickness = 30mm 
    -cut 8: 30 deg R angled slice, thickness = 30mm 
    -cut 9: in hand cut (90 deg), mid-cut 5
    -cut 10: in hand cut (90 deg), mid-cut 9
    '''
    deg30L_rotmat = np.array([
    [ 1.0,  0.0,   0.0], 
    [ 0.0, 0.866,  -0.5],
    [ 0.0,   0.5, 0.866]
    ])
    deg30R_rotmat = np.array([
    [ 1.0,  0.0,   0.0], 
    [ 0.0, 0.866,  0.5],
    [ 0.0,   -0.5, 0.866]
    ]) 
    deg90_rotmat = np.array([
    [ 0.0,  -1.0,   0.0], 
    [ 1.0, 0.0,  0.0],
    [ 0.0,   0.0, 1.0]
    ])
    # need to add in kinect/realsense/audio data saving

    # first do straight back and forth cuts
    thicknesses = [0.003, 0.005, 0.01, 0.02, 0.05]
    for slice_thickness in thicknesses:  
        delta_gripper_pose= RigidTransform(rotation=np.eye(3), translation=np.array([0,-slice_thickness,0]), from_frame='franka_tool',to_frame='franka_tool')      
        relative_vertical_height = 0.1
        object_vertical_contact_forces = [10.0, 10.0, 8.0, 10.0, 10.0, 10.0]
        num_dmps = 6
      
        for i in range(num_dmps):                           
            fa.execute_pose_dmp(pose_dmp_info, duration=4.8, cartesian_impedance=[3000, 3000, 3000, 10, 300, 300])            
            fa2.goto_pose_delta(delta_gripper_pose,duration=1.0)


    # do angled cuts
