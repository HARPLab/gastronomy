'''
experiment script for using REPS to learn HL skill parameters === xyz axes controller combos for different cutting skills/food types
'''

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import *

from tqdm import trange
from rl_utils import reps
import time 
import copy

def execute_sampled_action_and_get_reward(sample, starting_position, traject_time, dmp_wts_file, cut_type, S):
    # Calculate dmp trajectory  
    # traject_time = args.dmp_traject_time   # define length of dmp trajectory  
    # Load dmp traject params
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(dmp_wts_file)
    dmp_traj.parse_dmp_params_dict()

    # Define starting position 
    start_pose = fa.get_pose()
    starting_rotation = start_pose.rotation
    y0 = start_pose.translation 
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)

    # sampling info for sending msgs via ROS
    dt = 0.01 
    T = traject_time
    ts = np.arange(0, T, dt)
    N = len(ts)

    # downsample dmp traject 
    downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
    target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
    
    #import pdb; pdb.set_trace()
    # define controller stiffnesses
    position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    # set pitch axis cartesian gain to be sampled value
    position_kps_cart[-2] = cart_pitch_stiffness            
    force_kps_cart = [0.1] * 6
    position_kps_joint = FC.DEFAULT_K_GAINS
    force_kps_joint = [0.1] * 7

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)
    n_times = 1
    rospy.loginfo('Publishing HFPC trajectory w/ cartesian gains...')
    
    current_ht = fa.get_pose().translation[2]
    dmp_num = 0             

    # sample from gaussian to get dmp weights for this execution       
    #import pdb; pdb.set_trace()     
    dmp_num = 0            
    peak_z_forces_all_dmps, x_mvmt_all_dmps, forw_vs_back_x_mvmt_all_dmps = [], [], []# sum of abs (+x/-x mvmt)  
    y_mvmt_all_dmps, peak_y_force_all_dmps, z_mvmt_all_dmps, upward_z_mvmt_all_dmps, upward_z_penalty_all_dmps = [], [], [], [], []
    total_cut_time_all_dmps = 0          
    while current_ht > 0.023:   
        # start FP skill
        fa.run_dynamic_force_position(duration=T *100000000000000000, buffer_time = 3, 
                                    force_thresholds = [60.0, 60.0, 60.0, 30.0, 30.0, 30.0],
                                    S=S, use_cartesian_gains=True,
                                    position_kps_cart=position_kps_cart,
                                    force_kps_cart=force_kps_cart, block=False)

        print('starting dmp', dmp_num) 
        robot_positions = np.zeros((0,3))
        robot_forces = np.zeros((0,6))       
        init_time = rospy.Time.now().to_time()
        for i in trange(N * n_times):
            t = i % N                 
            timestamp = rospy.Time.now().to_time() - init_time
            #NOTE: format of pose sent is: 1x16 Transform matrix 
            
            traj_gen_proto_msg = ForcePositionSensorMessage(
                id=i, timestamp=timestamp, seg_run_time=dt,
                pose=target_poses[t],
                force=target_force
            )

            fb_ctrlr_proto = ForcePositionControllerSensorMessage(
                id=i, timestamp=timestamp,
                position_kps_cart=position_kps_cart,
                force_kps_cart=force_kps_cart,
                selection=S
            )

            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.FORCE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    fb_ctrlr_proto, SensorDataMessageType.FORCE_POSITION_GAINS)
                )
            
            robot_positions = np.vstack((robot_positions, fa.get_pose().translation.reshape(1,3)))
            robot_forces = np.vstack((robot_forces, fa.get_ee_force_torque().reshape(1,6)))
            
            pub.publish(ros_msg)
            rate.sleep() 

        # stop skill here w/ proto msg
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        pub.publish(ros_msg)

        # calc stats from dmp
        cut_time = rospy.Time.now().to_time() - init_time
        peak_y_force = np.max(np.abs(robot_forces[:,1]))
        peak_z_force = np.max(np.abs(robot_forces[:,2]))

        forward_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[0,0])))
        backward_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[-1,0])))
        total_x_mvmt = forward_x_mvmt + backward_x_mvmt

        forward_y_mvmt = (np.max(np.abs(robot_positions[:,1]) - np.abs(robot_positions[0,1])))
        backward_y_mvmt = (np.max(np.abs(robot_positions[:,1]) - np.abs(robot_positions[-1,1])))
        total_y_mvmt = forward_y_mvmt + backward_y_mvmt               
        
        if cut_type == 'normal' or cut_type == 'scoring':
            upward_z_mvmt = np.max(robot_positions[:,2]) - robot_positions[0,2]
            up_z_mvmt = np.abs(robot_positions[-1,2]) - np.min(np.abs(robot_positions[:,2])) 
            down_z_mvmt = np.abs(robot_positions[0,2]) - np.min(np.abs(robot_positions[:,2]))
            total_z_mvmt = up_z_mvmt + down_z_mvmt

        elif cut_type == 'pivchop':
            if (robot_positions[-1,2]-robot_positions[0,2]) > 0.02:
                upward_z_penalty = (robot_positions[-1,2]-robot_positions[0,2])
            else:
                upward_z_penalty = 0          
            up_z_mvmt = np.abs(robot_positions[-1,2]) - np.min(np.abs(robot_positions[:,2])) 
            down_z_mvmt = np.abs(robot_positions[0,2]) - np.min(np.abs(robot_positions[:,2]))
            total_z_mvmt = up_z_mvmt + down_z_mvmt
            diff_up_down_z_mvmt = np.abs(up_z_mvmt - down_z_mvmt)

        # save to buffers 
        total_cut_time_all_dmps += cut_time
        peak_z_forces_all_dmps.append(peak_z_force)
        x_mvmt_all_dmps.append(total_x_mvmt)
        y_mvmt_all_dmps.append(total_y_mvmt)
        peak_y_force_all_dmps.append(peak_y_force)           
        z_mvmt_all_dmps.append(total_z_mvmt)

        if args.cut_type == 'normal' or args.cut_type =='scoring':
            upward_z_mvmt_all_dmps.append(upward_z_mvmt)

        elif args.cut_type == 'pivchop':
            upward_z_penalty_all_dmps.append(upward_z_penalty)
        
        np.savez(work_dir + '/' + 'forces_positions/' + '_ep_'+str(sample) + '_trial_info_'+str(dmp_num)+'.npz', robot_positions=robot_positions, \
            robot_forces=robot_forces)
        
        completed_cut = input('cut complete? (0-n, 1-y, 2-cannot complete): ')
        #completed_cut = '2'

        while completed_cut not in ['0', '1', '2']:
            completed_cut = input('please enter valid answer. cut complete? (0/1/2): ') 

        if completed_cut == '1': 
            #fa.stop_skill()
            break

        elif completed_cut == '2': 
            # if cut can't be completed, give very high penalty for time 
            total_cut_time_all_dmps = 200
            #fa.stop_skill()
            break

        elif completed_cut == '0':
            current_ht = fa.get_pose().translation[2]
            print('current_ht', current_ht)
            dmp_num += 1  

            # calculate new dmp traject based on current position
            y0 = fa.get_pose().translation
            # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
            dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
                        
            downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
            target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
        
    
    # After finishing set of dmps for a full slice - calculate avg reward here
    # pause to let skill fully stop
    time.sleep(1.5)

    avg_peak_y_force = np.max(peak_y_force_all_dmps)
    avg_peak_z_force = np.max(peak_z_forces_all_dmps)
    avg_x_mvmt = np.max(x_mvmt_all_dmps)
    avg_y_mvmt = np.max(y_mvmt_all_dmps)
    avg_z_mvmt = np.max(z_mvmt_all_dmps)

    if cut_type == 'normal' or cut_type == 'scoring':
        avg_upward_z_mvmt = np.max(upward_z_mvmt_all_dmps)
        avg_upward_z_penalty = avg_upward_z_mvmt
    elif cut_type == 'pivchop':
        avg_upward_z_penalty = np.max(upward_z_penalty_all_dmps)

    deviation_actual_vs_demo_downward_z_mvmt = np.abs((downsmpled_dmp_traject[:,2][0] - min(downsmpled_dmp_traject[:,2])) - (robot_positions[:,2][0] - min(robot_positions[:,2])))
    deviation_actual_vs_demo_x_mvmt = np.abs((downsmpled_dmp_traject[:,0][0] - min(downsmpled_dmp_traject[:,0])) - (robot_positions[:,0][0] - min(robot_positions[:,0])))

    # new reward w/ added penalty for deviation from demo traject
    if cut_type == 'pivchop':
        reward = -0.1*avg_peak_y_force -0.15*avg_peak_z_force - 10*avg_x_mvmt -100*avg_y_mvmt - 10*avg_z_mvmt \
            -100*avg_upward_z_penalty -0.2*total_cut_time_all_dmps - 50000*np.square(deviation_actual_vs_demo_downward_z_mvmt)
    
    elif cut_type == 'scoring':
        reward = -0.1*avg_peak_y_force -0.15*avg_peak_z_force - 10*avg_x_mvmt -100*avg_y_mvmt - 10*avg_z_mvmt \
            -100*avg_upward_z_penalty -0.2*total_cut_time_all_dmps - 50000*np.square(deviation_actual_vs_demo_x_mvmt)
    
    else:
        # original reward for LL param optimiz
        reward = -0.1*avg_peak_y_force -0.15*avg_peak_z_force - 10*avg_x_mvmt -100*avg_y_mvmt - 10*avg_z_mvmt \
                -100*avg_upward_z_penalty -0.2*total_cut_time_all_dmps 
    
    # save reward to buffer
    print('Sample: %i Reward: '%(sample), reward)

    # reset to starting cut position            
    new_position = copy.deepcopy(starting_position)
    new_position.translation[1] = fa.get_pose().translation[1]
    fa.goto_pose(new_position, duration=5, use_impedance=False)

    # move over a bit (y dir)       
    y_shift = 0.004 #float(input('enter how far to shift in y dir (m): '))
    move_over_slice_thickness = RigidTransform(translation=np.array([0.0, y_shift, 0.0]),
        from_frame='world', to_frame='world') 
    # move_over_slice_thickness = RigidTransform(translation=np.array([0.0, 0.005, 0.0]),from_frame='world', to_frame='world') 
    fa.goto_pose_delta(move_over_slice_thickness, duration=3, use_impedance=False)

    # move down to contact
    import pdb; pdb.set_trace()
    move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
    from_frame='world', to_frame='world')   
    fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
    
    #import pdb; pdb.set_trace()
    return reward

def plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, save_dir, new_z_force, traject_time, \
    initial_dmp_weights_pkl_file, new_dmp_traject, y0):
    #original_dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(initial_dmp_weights_pkl_file)
    dmp_traj.parse_dmp_params_dict()
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    original_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    axes = ['x', 'y','z']   
    fig, ax = plt.subplots(3,1) 
    for i in range(3):
        ax[i].plot(np.arange(0, traject_time, 0.001), original_traject[:,i])
        ax[i].plot(np.arange(0, traject_time, 0.001), new_dmp_traject[:,i])           
        if i!=0:
            ax[i].set_title('Cartesian Position - '+str(axes[i]))
        else:     
            if new_z_force == 'NA':
                ax[i].set_title('Cartesian Position - '+str(axes[i]))
            else:
                ax[i].set_title('Cartesian Position - '+str(axes[i]) + ' ' + 'Downward z-force (N): '+str(new_z_force))
        ax[i].set_ylabel('Position (m)')
        ax[i].legend((axes[i] + '-original traject', axes[i] + '-new sampled traject'))
    
    ax[2].set_xlabel('Time (s)')
    plt.show()
    # save figure to working dir
    if os.path.isdir(work_dir + '/' + 'dmp_traject_plots'):
        fig.savefig(work_dir + '/' + 'dmp_traject_plots' + '/sampledDMP_' + 'epoch_'+str(epoch) + '_ep_'+str(sample)+'.png')

def plot_updated_policy_mean_traject(work_dir, position_dmp_weights_file_path, epoch, dmp_traject_time, control_type_z_axis, init_dmp_info_dict, \
    initial_wts, REPS_updated_mean):
    if control_type_z_axis == 'force':                
        new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
        new_z_force = REPS_updated_mean[-2]
    elif control_type_z_axis == 'position':
        new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],REPS_updated_mean[7:14])),axis=1)
        new_z_force = 0

    # Save new weights to dict
    data_dict = {
        'tau':           init_dmp_info_dict['tau'],
        'alpha':         init_dmp_info_dict['alpha'],
        'beta':          init_dmp_info_dict['beta'],
        'num_dims':      init_dmp_info_dict['num_dims'],
        'num_basis':     init_dmp_info_dict['num_basis'],
        'num_sensors':   init_dmp_info_dict['num_sensors'],
        'mu':            init_dmp_info_dict['mu'],
        'h':             init_dmp_info_dict['h'],
        'phi_j':         init_dmp_info_dict['phi_j'],
        'weights':       new_weights.tolist(),                
        }

    # save new sampled weights to pkl file
    weight_save_file = os.path.join(work_dir, 'meanWeightsUpdatedPol' + '.pkl')
    save_weights(weight_save_file, data_dict)

    # Calculate dmp trajectory             
    traject_time = dmp_traject_time   # define length of dmp trajectory  
    # Load dmp traject params
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(weight_save_file)
    dmp_traj.parse_dmp_params_dict()

    # Define starting position 
    start_pose = fa.get_pose()
    starting_rotation = start_pose.rotation
    y0 = start_pose.translation 
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    # check new dmp sampled wt trajectory vs original
    sample = 0
    plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, work_dir, new_z_force, traject_time, \
        position_dmp_weights_file_path, dmp_traject, y0)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cut_type', type=str, default = 'normal', help ='options are: normal, pivchop, scoring') # options: 'normal', 'pivchop', 'scoring'
    parser.add_argument('--dmp_traject_time', '-t', type=int, default = 5)  
    parser.add_argument('--param_sampling_var', type=float, default = 0.3)
    parser.add_argument('--num_epochs', '-e', type=int, default = 5)  
    parser.add_argument('--num_samples', '-s', type=int, default = 20)    
    # parser.add_argument('--data_savedir', '-d', type=str, 
    #     default='/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-controller-combo-exps/potato/pivChop/')
    parser.add_argument('--food_type', type=str, default = 'potato') 
    parser.add_argument('--data_savedir', '-d', type=str, 
        default='/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-controller-combo-exps/')
    parser.add_argument('--exp_num', '-n', type=int)
    parser.add_argument('--start_from_previous', '-sfp', type=bool, default=False)
    parser.add_argument('--previous_datadir', '-pd', type=str)
    parser.add_argument('--prev_epochs_to_calc_pol_update', '-num_prev_epochs', type=int, default = 1, help='num \
        previous epochs of data to use to calculate REPS policy update')   
    parser.add_argument('--starting_epoch_num', '-start_epoch', type=int, default = 0)
    parser.add_argument('--starting_sample_num', '-start_sample', type=int, default = 0)
    args = parser.parse_args()

    # create folders to save data
    if not os.path.isdir(args.data_savedir + args.food_type + '/' + args.cut_type + '/'):
        createFolder(args.data_savedir + args.food_type + '/' + args.cut_type + '/')
    args.data_savedir = args.data_savedir + args.food_type + '/' + args.cut_type + '/'

    if not os.path.isdir(args.data_savedir + 'exp_' + str(args.exp_num)):
        createFolder(args.data_savedir + 'exp_' + str(args.exp_num))

    work_dir = args.data_savedir + 'exp_' + str(args.exp_num)

    if not os.path.isdir(work_dir + '/' + 'all_polParamRew_data'):
        createFolder(work_dir + '/' + 'all_polParamRew_data')   
    if not os.path.isdir(work_dir + '/' + 'forces_positions'):
        createFolder(work_dir + '/' + 'forces_positions')

    # define DMP weight file based on type of cut desired 
    if args.cut_type == 'normal':
        dmp_wts_file = '/home/sony/092420_normal_cut_dmp_weights_zeroY.pkl'
        # more angled to sharp knife - normal cut
        knife_orientation = np.array([[0.0,   0.9805069,  -0.19648464],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.19648464,  -0.9805069]])

    elif args.cut_type == 'pivchop':
        dmp_wts_file = '/home/sony/raw_IL_trajects/Jan-2021/011321_piv_chop_potato_position_weights_zeroXY.pkl' 
        # metal knife (26 deg tilt forward) - pivchop
        knife_orientation = np.array([[0.0,   0.8988,  -0.4384],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.4384,  -0.8988]])

    elif args.cut_type == 'scoring':
        dmp_wts_file = '/home/sony/raw_IL_trajects/Jan-2021/011321_scoring_potato_position_weights_zeroYZ.pkl' 
        # metal knife (26 deg tilt forward) - pivchop
        knife_orientation = np.array([[0.0,   0.8988,  -0.4384],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.4384,  -0.8988]])    

    
    position_dmp_pkl = open(dmp_wts_file,"rb")
    init_dmp_info_dict = pickle.load(position_dmp_pkl)

    print('Starting robot')
    fa = FrankaArm()
    
    reset_joint_positions = [ 0.02846037, -0.51649966, -0.12048514, -2.86642333, -0.05060268,  2.30209197, 0.7744833 ]
    fa.goto_joints(reset_joint_positions)    

    # go to initial cutting pose
    starting_position = RigidTransform(rotation=knife_orientation, \
        translation=np.array([0.424, 0.102, 0.10]), #[0.486, 0.069, 0.12] - piv chop and normal # scoring
        from_frame='franka_tool', to_frame='world')    
    fa.goto_pose(starting_position, duration=5, use_impedance=False)

    # move down to contact food
    move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
    from_frame='world', to_frame='world')   
    fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, 
        force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
    
    # initialize LL params from IL - these will be fixed for this experiment (only trying to learn HL params here)
    pos_dmp_wts = np.array(init_dmp_info_dict['weights'])
    
    if args.cut_type == 'normal' or args.cut_type == 'scoring':
        cart_pitch_stiffness = 200 #400 
        target_force = [0, 0, -10, 0, 0, 0] # assuming zero force in y, downward force in z. TODO: how to define x axis force? 
    elif args.cut_type == 'pivchop':
        cart_pitch_stiffness = 20 #50
        target_force = [0, 0, -10, 0, 0, 0] # assuming zero force in y, downward force in z. TODO: how to define x axis force?   
    
    # UCB 
    possible_actions = [[1,1,1,1,1,1],[1,1,0,1,1,1],[1,0,1,1,1,1],[0,1,1,1,1,1],[0,0,0,1,1,1],[0,0,1,1,1,1],[0,1,0,1,1,1],[1,0,0,1,1,1]] # all 8 controller combos (S)
    nA = len(possible_actions)
    sum_of_rews = [0]*nA
    total_rew = 0
    
    T = args.num_samples #40
    actions_selected = []    
    num_selections = [0]*nA
    num_selections = [1,1,1,1,1,1,1,1] # initialize to 1 to avoid numerical issues

    # sampling 
    for t in range(0, T):
        print('timestep ', t)
        ad = 0       
        avg_rew = np.array(sum_of_rews)/np.array(num_selections)
        best_action_idx = np.argmax(avg_rew + math.sqrt(2*math.log(t+1))/np.array(num_selections))
        
        # select the best action
        actions_selected.append(best_action_idx)
        best_action = possible_actions[best_action_idx]
        print('S', best_action)
        reward = execute_sampled_action_and_get_reward(t, starting_position, args.dmp_traject_time, dmp_wts_file, args.cut_type, best_action)
        num_selections[best_action_idx]+=1
        sum_of_rews[best_action_idx] += reward
        total_rew+=reward 

    avg_rew = np.array(sum_of_rews)/np.array(num_selections)
    print('avg_rew', avg_rew)
    print('action w highest avg rew ', possible_actions[np.argmax(avg_rew)])          
    np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'sumRews' + '.npy'),np.array(sum_of_rews)) 
    np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'numSelections' + '.npy'),np.array(num_selections))
    np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'avgRews' + '.npy'),avg_rew)  



    