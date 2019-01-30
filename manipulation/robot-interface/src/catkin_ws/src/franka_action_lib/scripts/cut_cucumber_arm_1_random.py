#! /usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib')
import rospy
import actionlib
import pickle
import argparse
import numpy as np

from franka_action_lib.msg import ExecuteSkillAction, ExecuteSkillGoal

from frankapy.skill_list import *

def feedback_callback(feedback):
    print(feedback)


class CutCucumberSkill(object):
    INITIAL_POSITION = [-0.0372113,0.999006,0.0241583,0,0.998733,0.0379922,
                        -0.0327119,0,-0.0335979,0.0229109,-0.999173,0,0.458702,
                        -0.243681,0.245551,1]

    POSITION_ABOVE_CUTTING_BOARD = [0.00428579,0.999112,-0.0416815,0,0.999585,
                                    -0.00545372,-0.0279469,0,-0.02815, -0.0415,
                                    -0.99874,0,0.502556,-0.175819,0.141577,1]

    MOVE_TO_CUTTING_BOARD_POSITION = [0.00338211,0.99924,-0.0385855,0,0.999849,
                                      -0.00401365,-0.0163014,0,-0.0164442,
                                      -0.0385253,-0.999122,0,0.496864,
                                      -0.176568,0.0293225,1]
    MOVE_TO_CUCUMBER_POSITION = [0.00193678,0.999977,0.00475145,0,0.9999,
                                 -0.00199989,0.0133132,0,0.0133227,0.00472528,
                                 -0.9999,0,0.517015,0.04119,0.024956,1]

    SLICE_THICKNESS = 0.007
    FIRST_SLICE_THICKNESS = 0.01

    RELATIVE_MOTION_TO_CONTACT_FOR_CUTTING = 0.08

    # Quaternion helpers
    IDENTITY_QUATERNION = [1., 0., 0., 0.]

    # Time helpers
    RANDOM_EXPLORATION_TIME = 1.0

    def __init__(self, cutting_knife_location_x):
        self.cutting_knife_location_x = cutting_knife_location_x
        self.skill_id = 0

    def execute_skill(self, skill, client):
        goal = skill.create_goal()
        print(goal)
        client.send_goal(goal, feedback_cb=lambda x: skill.feedback_callback(x))
        done = client.wait_for_result(rospy.Duration.from_sec(5.0))
        while not rospy.is_shutdown() and done != True:
            print(goal)
            done = client.wait_for_result(rospy.Duration.from_sec(5.0))
        print(client.get_result())
        self.skill_id = self.skill_id + 1

    def create_skill_for_class(self, klass, description):
        skill = klass(
                skill_description=
                description 
                + ' type: {}, '.format(klass.__name__)
                + ' id: {}'.format(self.skill_id))
        return skill

    def get_move_left_skill(self, distance_in_m, desc=''):
        skill = self.create_skill_for_class(
                ArmRelativeMotionWithDefaultSensorSkill,
                desc)
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])
        skill.add_relative_motion_with_quaternion(
                3.0,
                [0., -distance_in_m, 0.],
                CutCucumberSkill.IDENTITY_QUATERNION)
        skill.add_feedback_controller_params([600, 50])
        skill.add_termination_params([1.0])
        return skill

    def create_skill_to_move_to_cucumber(self, desc=''):
        ''' Move left to contact cucumber '''
	skill = self.create_skill_for_class(
		ArmMoveToGoalContactWithDefaultSensorSkill,
		desc)
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])
        skill.add_trajectory_params(
                [3.0] + CutCucumberSkill.MOVE_TO_CUCUMBER_POSITION)
        skill.add_feedback_controller_params([600, 50])
        skill.add_contact_termination_params(1.0, [10.0] * 6, [10.0] * 6)
        return skill

    def add_random_exploration(
            self,
            time,
            position_delta,
            lower_force_thresholds_accel=[10.0] * 6,
            lower_force_thresholds_nominal=[10.0] * 6,
            description=''):
        skill = self.create_skill_for_class(
            ArmRelativeMotionToContactWithDefaultSensorSkill,
            description)
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])
        skill.add_traj_params_with_quaternion(
                time,
                position_delta,
                CutCucumberSkill.IDENTITY_QUATERNION)
        skill.add_controller_stiffness_params(600, 50)
        skill.add_contact_termination_params(
                1.0,
                lower_force_thresholds_accel,
                lower_force_thresholds_nominal)
        return skill

    def add_random_x_exploration(
            self,
            time,
            x_delta,
            lower_force_thresholds_accel=[10.0] * 6,
            lower_force_thresholds_nominal=[10.0] * 6,
            description=''):
        return self.add_random_exploration(
                time,
                [x_delta, 0., 0.],
                lower_force_thresholds_accel,
                lower_force_thresholds_nominal,
                description=description)

    def add_random_y_exploration(
            self,
            time,
            y_delta,
            lower_force_thresholds_accel=[10.0] * 6,
            lower_force_thresholds_nominal=[10.0] * 6,
            description=''):
        return self.add_random_exploration(
                time,
                [0., y_delta, 0.],
                lower_force_thresholds_accel,
                lower_force_thresholds_nominal,
                description=description)

    def add_random_z_exploration(
            self,
            time,
            z_delta,
            lower_force_thresholds_accel=[10.0] * 6,
            lower_force_thresholds_nominal=[10.0] * 6,
            description=''):
        return self.add_random_exploration(
                time,
                [0., 0., z_delta],
                lower_force_thresholds_accel,
                lower_force_thresholds_nominal,
                description=description)
    
    def run_random_exploration_skills(self, time, delta_movement, 
                                      desc_prefix=''):
        # Add random exploration to know that you're on the cutting board
        d = delta_movement
        d_suffix = '_pos_{:.3f}_time_{:.3f}'.format(d, time)
        skill = cut_cucumber_skill.add_random_x_exploration(
                time, d, 
                description=desc_prefix+'_x_1'+d_suffix)
        self.execute_skill(skill, client)
        skill = cut_cucumber_skill.add_random_x_exploration(
                time, -2 * d,
                description=desc_prefix+'_x_2'+d_suffix)
        self.execute_skill(skill, client)
        skill = cut_cucumber_skill.add_random_x_exploration(
                time, d,
                description=desc_prefix+'_x_3'+d_suffix)
        self.execute_skill(skill, client)

        skill = cut_cucumber_skill.add_random_y_exploration(
                time, d,
                description=desc_prefix+'_y_1'+d_suffix)
        self.execute_skill(skill, client)
        skill = cut_cucumber_skill.add_random_y_exploration(
                time, -2 * d,
                description=desc_prefix+'_y_2'+d_suffix)
        self.execute_skill(skill, client)
        skill = cut_cucumber_skill.add_random_y_exploration(
                time, d,
                description=desc_prefix+'_y_3'+d_suffix)
        self.execute_skill(skill, client)

        skill = cut_cucumber_skill.add_random_z_exploration(
                time, d,
                description=desc_prefix+'_z_1'+d_suffix)
        self.execute_skill(skill, client)
        skill = cut_cucumber_skill.add_random_z_exploration(
                time, -2 * d,
                description=desc_prefix+'_z_2'+d_suffix)
        self.execute_skill(skill, client)
        skill = cut_cucumber_skill.add_random_z_exploration(
                time, d,
                description=desc_prefix+'_z_3'+d_suffix)
        self.execute_skill(skill, client)


if __name__ == '__main__':
    rospy.init_node('example_execute_skill_action_client')
    client = actionlib.SimpleActionClient(
            '/execute_skill_action_server_node/execute_skill',
            ExecuteSkillAction)
    client.wait_for_server()

    parser = argparse.ArgumentParser(description='Joint DMP Skill Example')
    parser.add_argument('--filename', required=True, help='filename with dmp weights')
    args = parser.parse_args()

    file = open(args.filename,"rb")
    dmp_info = pickle.load(file)
    cutting_knife_location_x = 0.5232
    cut_cucumber_skill = CutCucumberSkill(cutting_knife_location_x)

    skill = cut_cucumber_skill.create_skill_for_class(
            ArmMoveToGoalWithDefaultSensorSkill,
            'move_to_initial_position')
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    # Run Time (1) and Desired End Effector Pose(16)
    skill.add_trajectory_params([3.0] + CutCucumberSkill.INITIAL_POSITION)
    # translational stiffness, rotational stiffness
    skill.add_feedback_controller_params([600, 50])
    skill.add_buffer_time_for_termination(1.0)
    cut_cucumber_skill.execute_skill(skill, client)

    # Move to designated position above the cutting board
    skill = cut_cucumber_skill.create_skill_for_class(
            ArmMoveToGoalWithDefaultSensorSkill,
            'move_above_cutting_board')
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params(
            [3.0] + CutCucumberSkill.POSITION_ABOVE_CUTTING_BOARD)
    skill.add_feedback_controller_params([600, 50])
    skill.add_buffer_time_for_termination(1.0)
    cut_cucumber_skill.execute_skill(skill, client)

    # Move down to contact cutting board
    skill = cut_cucumber_skill.create_skill_for_class(
            ArmMoveToGoalWithDefaultSensorSkill,
            'move_onto_cutting_board')
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params(
            [3.0] + CutCucumberSkill.MOVE_TO_CUTTING_BOARD_POSITION)
    skill.add_feedback_controller_params([600, 50])
    skill.add_buffer_time_for_termination(1.0)
    cut_cucumber_skill.execute_skill(skill, client)

    # ==== Begin Random exploration ====
    # Add random exploration to know that you're on the cutting board
    cut_cucumber_skill.run_random_exploration_skills(
            0.2, 0.005, desc_prefix='random_exploration_on_cutting_board')
    # ==== End ====

    # Move left to contact cucumber
    skill = cut_cucumber_skill.create_skill_for_class(
             ArmMoveToGoalContactWithDefaultSensorSkill,
             'move_left_to_contact_cucumber')
    skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
    skill.add_trajectory_params(
            [3.0] + CutCucumberSkill.MOVE_TO_CUCUMBER_POSITION)
    skill.add_feedback_controller_params([600, 50])

    skill.add_contact_termination_params(
            1.0,
            [10.0,3.0,10.0,10.0,10.0,10.0],
            [10.0,3.0,10.0,10.0,10.0,10.0])
    cut_cucumber_skill.execute_skill(skill, client)

    # ==== Begin Random exploration ====
    # Add random exploration to know that you're on the cutting board
    cut_cucumber_skill.run_random_exploration_skills(0.2, 0.005,
            desc_prefix='random_exploration_next_to_cucumber')
    # ==== End ====


    num_slices_to_cut = 4

    for slice_idx in range(num_slices_to_cut):
        # Move up above the cucumber
        skill = cut_cucumber_skill.create_skill_for_class(
            ArmRelativeMotionWithDefaultSensorSkill,
            'move_up_above_cucumber_{}'.format(slice_idx))
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
        skill.add_relative_motion_with_quaternion(
                1.0,
                [0., 0., CutCucumberSkill.RELATIVE_MOTION_TO_CONTACT_FOR_CUTTING],
                CutCucumberSkill.IDENTITY_QUATERNION)

        skill.add_feedback_controller_params([600, 50])
        skill.add_termination_params([1.0])
        cut_cucumber_skill.execute_skill(skill, client)

        slice_thickness = CutCucumberSkill.SLICE_THICKNESS \
                if slice_idx > 0 else CutCucumberSkill.FIRST_SLICE_THICKNESS
        # Move left above the cucumber
        skill = cut_cucumber_skill.create_skill_for_class(
            ArmRelativeMotionWithDefaultSensorSkill,
            'move_left_to_cut_slice_{}'.format(slice_idx))
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
        skill.add_relative_motion_with_quaternion(
                1.0,
                [0., slice_thickness, 0.],
                CutCucumberSkill.IDENTITY_QUATERNION)

        skill.add_feedback_controller_params([600, 50])
        skill.add_termination_params([1.0])
        cut_cucumber_skill.execute_skill(skill, client)

        # Move to contact
        skill = cut_cucumber_skill.create_skill_for_class(
                ArmRelativeMotionToContactWithDefaultSensorSkill,
                'move_onto_cucumber_to_cut_{}'.format(slice_idx))
        skill.add_initial_sensor_values([1, 3, 5, 7, 8])  # random
        skill.add_traj_params_with_quaternion(
                1.0,
                [0., 0., -CutCucumberSkill.RELATIVE_MOTION_TO_CONTACT_FOR_CUTTING],
                CutCucumberSkill.IDENTITY_QUATERNION)
        skill.add_controller_stiffness_params(600, 50)
        skill.add_contact_termination_params(1.0, [10.0] * 6, [10.0] * 6)
        cut_cucumber_skill.execute_skill(skill, client)
        # ==== Begin Random exploration ====
        # Add random exploration to know that you're on the cutting board
        cut_cucumber_skill.run_random_exploration_skills(
                0.2, 0.005,
                desc_prefix='random_exploration_on_cucumber_{}'.format(slice_idx))
        # ==== End ====

        # Start DMP cutting for 3 times
        skill = cut_cucumber_skill.create_skill_for_class(
                JointPoseWithDefaultSensorSkill,
                'cut_dmp')
        skill.add_initial_sensor_values(dmp_info['phi_j'])  # sensor values
        # y0 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        y0 = [-0.282, -0.189, 0.0668, -2.186, 0.0524, 1.916, -1.06273]
        # Run time, tau, alpha, beta, num_basis, num_sensor_values, mu, h, weights
        trajectory_params = [
                4.0, dmp_info['tau'], dmp_info['alpha'], dmp_info['beta'],
                float(dmp_info['num_basis']), float(dmp_info['num_sensors'])] \
                        + dmp_info['mu'] \
                        + dmp_info['h'] \
                        + y0 \
                        + np.array(dmp_info['weights']).reshape(-1).tolist()

        skill.add_trajectory_params(trajectory_params)
        skill.set_meta_skill_id(slice_idx+1)
        skill.set_meta_skill_type(1)
        skill.add_termination_params([1.0])
        num_of_dmps_to_run = 4
        for _ in range(num_of_dmps_to_run):
            cut_cucumber_skill.execute_skill(skill, client)

        skill = cut_cucumber_skill.get_move_left_skill(
                0.06, 
                desc='move_to_separate_cut_slice_{}'.format(slice_idx))
        cut_cucumber_skill.execute_skill(skill, client)

        skill = cut_cucumber_skill.create_skill_to_move_to_cucumber(
                desc='move_left_to_contact_cucumber_after_slice_{}'.format(slice_idx))
        cut_cucumber_skill.execute_skill(skill, client)
