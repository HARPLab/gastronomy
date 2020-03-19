import numpy as np
import argparse
import copy
import os
import pickle
import json
import pprint
import cv2
import shutil
import logging
import math

from utilities import vrep_utils as vu
from utilities.math_utils import Sat3D, sample_from_edges, are_position_similar
from utilities.math_utils import sample_exactly_around_edge
from utilities.math_utils import sample_exactly_outside_edge
from sim.sim_vrep.robot_scene_3 import RobotVrepScene3
from sim.sim_vrep.robot_scene_1 import _array_to_str

from lib import vrep
from utils.colors import bcolors

from action_relation.dataloader.vrep_scene_geometry import VrepSceneGeometry
from action_relation.dataloader.octree_data import OctreePickleLoader

npu = np.random.uniform
npri = np.random.randint


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class RobotVrepScene4OrientChange(RobotVrepScene3):
    def __init__(self, args):
        super(RobotVrepScene3, self).__init__(args)

    def sample_edge_location_for_other_obj(self, anchor_args, anchor_pos,
        anchor_lb, anchor_ub, anchor_in_air, other_obj_size, outside_xyz, 
        sample_axes_reg):
        '''Sample point on one of the edges of the cuboid.
        
        sample_axes_reg: List of len 3 with values 0, 1, 2 in some order, 
            where 0: in, 1: edge, 2:out
        '''

        xs, ys, zs = other_obj_size

        pos_list = []
        region_xyz = [None, None, 'high']
        # if anchor_in_air:
        #     region_xyz[2] = 'low'

        all_info = {'sample_region_xyz': []}
        for i in range(3):
            # in
            if sample_axes_reg[i] == 0:
                pos_i = npu(anchor_lb[i] + 0.01, anchor_ub[i] - 0.01)
                all_info['sample_region_xyz'].append('middle')
            elif sample_axes_reg[i] == 1:
                # Sample exactly at the edge
                pos_i, info = sample_exactly_around_edge(
                    anchor_lb[i], 
                    anchor_ub[i],
                    other_obj_size[i]/2.0,
                    region=region_xyz[i])
                all_info['sample_region_xyz'].append(info['sample_region'])
            elif sample_axes_reg[i] == 2:
                # Sample exactly outside the edge. Threshold controls the distance
                # between the objects, i.e we sample uniformly from (0, threshold)
                # and add the object that much distance away.
                pos_i, info = sample_exactly_outside_edge(
                    anchor_lb[i],
                    anchor_ub[i],
                    0.10,  # threshold 
                    other_obj_size[i]/2.0,
                    region=region_xyz[i],
                    min_threshold=0.05)
                all_info['sample_region_xyz'].append(info['sample_region'])
            else:
                raise ValueError("Invalid sample_axes_reg value")
            
            pos_list.append(pos_i)
        
        assert len(pos_list) == 3
        return pos_list, all_info

    def get_other_obj_location_outside_anchor_cuboid(
        self, anchor_args, anchor_pos, anchor_lb, anchor_ub, anchor_in_air, 
        other_obj_size, min_x=None, min_y=None, min_z=None):
        '''Get position to move other object to which is outside anchor cuboid.

        anchor_lb: Absolute lower bound of bounding box.
        anchor_ub: Absolute upper bound of bounding box.

        min_x: Float. Minimum X position for the other object.
        min_z: Float. Minimum Z position for the other object.

        Return: Position to move other object to.
        '''
        xs, ys, zs = [(other_obj_size[i]+0.01)/2.0 for i in range(3)]

        # sample_axes_reg : 0: in, 1: edge, 2:out
        # Find atleast one axes on which we're out.
        valid_sample_types = ('3_out', '1_edge_1_in_1_out', '2_edge_1_out', 
                              '2_in_1_out')
        sample_type = valid_sample_types[npri(0, len(valid_sample_types))]
        sample_type = '2_in_1_out'
        if sample_type == '3_out':
            sample_axes_reg = [2, 2, 2]
        elif sample_type == '1_edge_1_in_1_out':
            sample_axes_reg = [0, 1, 2]
        elif sample_type == '2_edge_1_out':
            sample_axes_reg = [1, 1, 2]
        elif sample_type == '2_in_1_out':
            sample_axes_reg = [0, 0, 2]
        else:
            raise ValueError("Invalid sample type")
        # np.random.shuffle(sample_axes_reg)

        outside_xyz = []
        for ax in sample_axes_reg:
            outside_val = True if ax == 2 else False
            outside_xyz.append(outside_val)

        # Why do this? Because numpy booleans are not JSON-serializable.
        if np.any([True for x in sample_axes_reg if x == 1]):
            sample_on_edge = True
        else:
            sample_on_edge = False

        [x, y, z], sample_edge_info = self.sample_edge_location_for_other_obj(
            anchor_args, anchor_pos, anchor_lb, anchor_ub, anchor_in_air,
            other_obj_size, outside_xyz, sample_axes_reg,
            )
        
        # Action type to take
        valid_action_types = ['all', 'max_torque']
        if sample_type == '3_out':
            action_type = 'all'
        else:
            # With some prob (0.8) take only the actions that will result in
            # interaction between objects.
            if npu(0, 1) < 0.0:
                action_type = 'max_torque'
            else:
                action_type = 'all'
        assert action_type in valid_action_types

        info = {'sample_on_edge': sample_on_edge,
                'sample_axes_reg': sample_axes_reg,
                'sample_edge_info': sample_edge_info,
                'outside_xyz': outside_xyz,
                'anchor_in_air': anchor_in_air,
                'valid_action_types': valid_action_types,
                'action_type': action_type,
                'valid_sample_type': valid_sample_types,
                'sample_type': sample_type,
                }

        return [x, y, z], info 

    def get_waypoints_for_other_obj_pos(self, other_obj_pos, joint_pos, sample_info):
        joint_offsets = self.get_joint_offsets()
        # First add joint offsets to new_other_obj_pos
        shifted_other_obj_pos = [other_obj_pos[i] - joint_offsets[i] 
                                 for i in range(3)]

        # Going directly to new_other_obj_pos can involve collisions with the
        # anchor object, hence take the path where we have no intersection first.
        other_obj_waypoints = [[0, 0, 0], shifted_other_obj_pos]
        for i in range(3):
            if sample_info['outside_xyz'][i]:
                other_obj_waypoints[0][i] = shifted_other_obj_pos[i]
            else:
                other_obj_waypoints[0][i] = joint_pos[i]

        return other_obj_waypoints, joint_offsets
    
    def run_scene(self, scene_i, save_dir):
        self.scene_info = {}
        args = self.args
        action_relative_or_absolute = 'absolute'

        logging.info("Start sim: {}".format(scene_i))
        self.reset()
        self.step()

        # Generate anchor handle
        anchor_args = self.get_anchor_object_args(min_obj_size=0.02, 
                                                  max_obj_size=0.10)
        # Move object above air
        anchor_in_air = args.anchor_in_air
        if anchor_in_air:
            # Move the object up so that it is in the air.
            anchor_args['pos'][2] += npu(0.28, 0.3)

        anchor_handle = self.generate_object_with_args(anchor_args)
        self.handles_dict['anchor_obj'] = anchor_handle

        # Make sure that the the octree position is at the base of the 
        octree_pos = [anchor_args['pos'][0], anchor_args['pos'][1], 
                      anchor_args['pos'][2] + anchor_args['obj_size'][2]/2.0]
        self.update_octree_position(octree_pos) 

        # Move it slightly above to account for the fact that there can be
        # inaccuracies in real world voxels.
        if npu(0, 1) < 0.3:
            # Move the object up so that it is in the air.
            anchor_args['pos'][2] += npu(0.01, 0.02)

        for _ in range(4):
            self.step()
        anchor_pos = self.get_object_position(anchor_handle)
        logging.info("Anchor pos: {}".format(_array_to_str(anchor_pos)))

        # Move the robot to nearby (above) the anchor object.
        if anchor_in_air:
            # Place the other object down. If the anchor is in air then there
            # is no point in placing the other object above the anchor. Hence,
            # we always keep it below
            orig_joint_position = [
                anchor_pos[0], anchor_pos[1], max(anchor_pos[2]-0.2, 0), 0, 0, 0]
        else:
            orig_joint_position = [anchor_pos[0], anchor_pos[1], anchor_pos[2]+0.2, 0, 0, 0]
        self.set_all_joint_positions(orig_joint_position)

        # Since the joint positions now have to be below the object move it below.
        if anchor_in_air:
            for _ in range(10):
                self.step()

        # Generate the other object
        other_pos = [0.0, 0.0, 0.0]
        other_obj_args = self.get_other_object_args(other_pos, 
                                                    add_obj_z=False,
                                                    min_size=0.02, 
                                                    max_size=0.10)
        other_handle = self.generate_object_with_args(other_obj_args)
        for _ in range(5):
            self.step()
        
        init_joint_offsets = self.get_joint_offsets()

        # Move other obj to the right place
        anchor_abs_bb = self.get_absolute_bb(anchor_handle)
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        new_other_obj_pos, sample_info = self.get_other_obj_location_outside_anchor_cuboid(
            anchor_args, 
            anchor_pos,
            anchor_lb,
            anchor_ub,
            anchor_in_air,
            other_obj_args['obj_size'])
        if new_other_obj_pos is None:
            logging.info("Cannot place other obj near anchor.")
            return False

        self.scene_info['new_other_obj_pos'] = new_other_obj_pos
        self.scene_info['sample_info'] = sample_info

        # What sort of actions to take? Take all actions or only actions that
        # result in orientation change
        action_type = sample_info['action_type']
        actions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
        diagonal_actions = [
            [-1, 1, 0], [-1, 0, 1], [-1, -1, 0], [-1, 0, -1],
            [1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, 1]
        ]
        diagonal_actions_3d = [
                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
            ]
        actions += diagonal_actions
        actions += diagonal_actions_3d
        if action_relative_or_absolute == 'absolute':
            action_dist = 0.08
            actions = [[action_dist*x[0], action_dist*x[1], action_dist*x[2]]
                       for x in actions]
        if action_type == 'max_torque':
            axes_to_action_idx = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
            valid_actions = copy.deepcopy(actions)
            for axes_i, axes_out in enumerate(sample_info['outside_xyz']):
                if not axes_out:
                    assert len(axes_to_action_idx[axes_i]) == 2
                    valid_actions[axes_to_action_idx[axes_i][0]] = None
                    valid_actions[axes_to_action_idx[axes_i][1]] = None
                else:
                    # Why True? See below.
                    take_positive_actions_only = True
                    if take_positive_actions_only:
                        # Make the negative action None
                        valid_actions[axes_to_action_idx[axes_i][0]] = None
                    else:
                        # NOTE: Since we always take action equal to moving to the 
                        # center the following piece of logic is not required anymore,
                        # because we take only positive actions 
                        region = sample_info['sample_edge_info']['sample_region_xyz'][axes_i]
                        if region == 'low':
                            valid_actions[axes_to_action_idx[axes_i][0]] = None
                            assert new_other_obj_pos[axes_i]
                        elif region == 'high':
                            valid_actions[axes_to_action_idx[axes_i][1]] = None
                        elif region == 'middle':
                            raise ValueError("Invalid region: {}".format(region))
                        else:
                            raise ValueError("Invalid region: {}".format(region))
            
            logging.info("Filtered actions from {} to {}".format(
                len(actions), len([va for va in valid_actions if va is not None])))
            actions = valid_actions

        has_created_scene = True
        for a_idx, a in enumerate(actions):
            if a is None:
                continue

            self.scene_info[a_idx] = { 
                'action': a, 
                'action_type': action_type,
                'action_relative_or_absolute': action_relative_or_absolute
            }
            logging.info(bcolors.c_red(
                " ==== Scene {} action: {} ({}/{}) start ====".format(
                    scene_i, a, a_idx, len(actions))))

            if args.reset_every_action == 1 and not has_created_scene:
                anchor_handle = self.generate_object_with_args(anchor_args)
                self.handles_dict['anchor_obj'] = anchor_handle
                for _ in range(5):
                    self.step()
                self.set_all_joint_positions(orig_joint_position)
                for _ in range(5):
                    self.step()
                other_handle = self.generate_object_with_args(other_obj_args)
                self.handles_dict['other_obj'] = other_handle
                for _ in range(5):
                    self.step()
            elif args.reset_every_action == 0 and  has_created_scene:
                # First go to the previous position so that it's easy to go 
                # back to the original position.
                # if new_other_obj_pos is not None:
                #     self.set_prismatic_joints(new_other_obj_pos)
                #     for _ in range(10):
                #         self.step()

                # First go to rest position
                self.set_all_joint_positions(orig_joint_position)
                for _ in range(10):
                    self.step()

            joint_pos = self.get_joint_position()
            logging.info("Current joint pos: {}".format(_array_to_str(joint_pos)))

            new_other_obj_waypoints, joint_offsets = self.get_waypoints_for_other_obj_pos(
                new_other_obj_pos, joint_pos, sample_info)
            logging.info(bcolors.c_yellow(
                "Should move other obj\n"
                "    \t                to:  {}\n"
                "    \t     First move to:  {}\n"
                "    \t       Now move to:  {}\n"
                "    \t      joint offset:  {}\n".format(
                _array_to_str(new_other_obj_pos),
                _array_to_str(new_other_obj_waypoints[0]),
                _array_to_str(new_other_obj_waypoints[1]),
                _array_to_str(joint_offsets))))

            # Move to the first waypoint
            logging.info(bcolors.c_green("Move joints to: {}".format(
                _array_to_str(new_other_obj_waypoints[0]))))
            self.set_prismatic_joints(new_other_obj_waypoints[0])
            for _ in range(15):
                self.step()
            _, temp_obj_data = self.get_all_objects_info()
            logging.info(bcolors.c_cyan(
                "      \t    Curr other obj location:   {}\n"
                "      \t             Curr joint pos:   {}\n"
                "      \t      Curr joint target pos:   {}\n".format(
                    _array_to_str(temp_obj_data['other_pos']),
                    _array_to_str(temp_obj_data['joint_pos']),
                    _array_to_str(temp_obj_data['joint_target_pos']),
                )))

            logging.info(bcolors.c_green("Move joints to: {}".format(
                _array_to_str(new_other_obj_waypoints[1]))))
            self.set_prismatic_joints(new_other_obj_waypoints[1])
            for _ in range(15):
                self.step()

            new_joint_offsets = self.get_joint_offsets()
            diff_offsets = [new_joint_offsets[i] - joint_offsets[i] for i in range(3)]

            _, first_obj_data_dict = self.get_all_objects_info()
            obj_at_desired_location = are_position_similar(
                first_obj_data_dict['other_pos'], new_other_obj_pos)
            if not obj_at_desired_location:
                logging.error(bcolors.c_cyan(
                    "OBJECTS NOT at desired location!!\n"
                    "        \t          curr: {}\n"
                    "        \t       desired: {}\n"
                    "        \t  joint_T curr: {}\n"
                    "        \t   joint_T des: {}\n".format(
                        _array_to_str(first_obj_data_dict['other_pos']),
                        _array_to_str(new_other_obj_pos),
                        _array_to_str(first_obj_data_dict['joint_target_pos']),
                        _array_to_str(new_other_obj_waypoints[1]),
                    )))
                return False

            # Increase the pid values before taking the action.
            '''
            old_pid_values = self.get_joint_pid_values()
            self.set_joint_pid_values(0.1, 0.0, 0.0)
            new_pid_values = self.get_joint_pid_values()
            logging.debug("Old pid values: {}\n"
                          "new pid values: {}\n".format(
                            _array_to_str(old_pid_values),
                            _array_to_str(new_pid_values)
                         ))
            '''
            for _ in range(10):
                self.step()
            _, second_obj_data_dict = self.get_all_objects_info()
            obj_in_place = self.are_pos_and_orientation_similar(
                first_obj_data_dict, second_obj_data_dict)

            if not obj_in_place:
                logging.error(bcolors.c_cyan(
                    "OBJECTS STILL MOVING!! Will sample again."))
                # return False
            # if not self.are_objects_close(second_obj_data_dict):
            #     logging.error("Objects are not close. Will sample again.")
            #     return False

            self.scene_info[a_idx]['before_dist'] = self.get_object_distance()[-1]

            # save vision and octree info before taking action
            before_contact_info = self.get_contacts_info()
            # before_ft_data = self.read_force_torque_data()
            # voxels_before_dict = self.run_save_voxels_before(
            #     anchor_handle, other_handle)
            voxels_before_dict = {}
            self.toggle_recording_contact_data()
            self.toggle_record_ft_data()
            self.step()

            # Now perform the action
            # after_action_joint_pos = [joint_pos[0], joint_pos[1], anchor_pos[2], 0, 0, 0]
            # The following actions are taken if we take "absolute actions"
            if action_relative_or_absolute == 'absolute':
                if action_type == 'max_torque':
                    diff = []
                    for ai in range(3):
                        if anchor_args['pos'][ai]-new_other_obj_waypoints[-1][ai] >= 0:
                            diff.append(a[ai])
                        else:
                            diff.append(-a[ai])
                    after_action_joint_pos = [new_other_obj_waypoints[-1][i]+diff[i]
                                              for i in range(3)]
                else:
                    after_action_joint_pos = [new_other_obj_waypoints[-1][i] + a[i]
                                              for i in range(3)]
            elif action_relative_or_absolute == 'relative':
                # Should we move fixed distance or adaptive distnaces?
                diff = [a[i]*(anchor_args['pos'][i]-new_other_obj_waypoints[-1][i]) 
                        for i in range(3)]
                after_action_joint_pos = [new_other_obj_waypoints[-1][i]+diff[i]
                                          for i in range(3)]
            else:
                raise ValueError(f"Invalid action {action_relative_or_absolute}")

            logging.info(f"After action pos: {after_action_joint_pos}")
            self.set_all_joint_positions(after_action_joint_pos + [0, 0, 0])

            for _ in range(25):
                self.step()
            
            # Stop the scene?
            # self.stop()

            _, third_obj_data_dict = self.get_all_objects_info()
            obj_in_place = self.are_pos_and_orientation_similar(
                first_obj_data_dict, second_obj_data_dict)
            self.debug_initial_final_position(
                first_obj_data_dict, second_obj_data_dict)
            self.scene_info[a_idx]['before'] = second_obj_data_dict 
            self.scene_info[a_idx]['after'] = third_obj_data_dict 
            if not obj_in_place: 
                logging.info("Objects changed position AFTER action.")
            self.scene_info[a_idx]['after_dist'] = self.get_object_distance()[-1]

            self.toggle_recording_contact_data()
            self.toggle_record_ft_data()
            _, recorded_contact_info = self.save_recorded_contact_data()
            recorded_ft_data = self.save_record_ft_data()
            ft_sensor_mean = np.mean(np.array(recorded_ft_data).reshape(-1, 6), 
                                     axis=0)
            self.scene_info[a_idx]['ft_sensor_mean'] = ft_sensor_mean.tolist()
            self.step()
            logging.info(bcolors.c_cyan("Mean force-torque val: {}".format(
                _array_to_str(ft_sensor_mean))))

            # Now save vision and octree info after taking action.
            after_contact_info = self.get_contacts_info()
            voxels_after_dict = self.run_save_voxels_after(
                anchor_handle, other_handle)
            
            for _ in range(10):
                self.step()
            
            self.save_scene_data(
                save_dir, a_idx, self.scene_info,
                voxels_before_dict,
                voxels_after_dict,
                before_contact_info,
                after_contact_info,
                recorded_contact_info,
                recorded_ft_data,
            )
            self.debug_initial_final_position(second_obj_data_dict, 
                                              third_obj_data_dict)
            before_contact_len = 0 if before_contact_info is None else before_contact_info.shape[0]
            after_contact_len = 0 if after_contact_info is None else after_contact_info.shape[0]
            logging.info("Contact len: before: {}, after: {}".format(
                before_contact_len, after_contact_len))
            logging.info(" ==== Scene {} action: {} Done ====".format(
                scene_i, a))
            if args.reset_every_action == 1:
                self.reset()
                has_created_scene = False

        return True


def main(args):
    scene = RobotVrepScene4OrientChange(args)
    num_scenes, total_try = 0, 0
    torque_th = 1e-3
    num_scenes_above_torque_th, total_scenes_actions = 0, 0
    while num_scenes < args.num_scenes:
        save_path = os.path.join(args.save_dir, '{:0>5d}'.format(num_scenes))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        status = scene.run_scene(num_scenes, save_path)
        if status:
            for action in scene.scene_info.keys():
                if type(action) is int:
                    ft_sensor_arr = np.array(
                        scene.scene_info[action]['ft_sensor_mean'])[3:6]
                    if np.any(np.abs(ft_sensor_arr)) > torque_th:
                        num_scenes_above_torque_th += 1
                    total_scenes_actions += 1
         

        total_try += 1
        if not status:
            shutil.rmtree(save_path)
            print("Incorrect scene, scenes finished: {}, succ: {:.2f}"
                  "  \t  torque_above_th: [{}/{}]".format(
                      num_scenes, float(num_scenes)/total_try,
                      num_scenes_above_torque_th, total_scenes_actions))
        else:
            num_scenes += 1
            print("Correct scene: scenes finished: {}, succ: {:.2f}"
                  "  \t  torque_above_th: [{}/{}]".format(
                      num_scenes, float(num_scenes)/total_try,
                      num_scenes_above_torque_th, total_scenes_actions))
    scene.stop_and_finish()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Create on scene in vrep")
    parser.add_argument('--port', type=int, default=19997,
                        help='Port to listen to vrep.')
    parser.add_argument('--num_scenes', type=int, default=10,
                        help='Number of scenes to generate.')
    parser.add_argument('--save_dir', type=str, default='/tmp/whatever',
                        help='Save results in dir.')
    parser.add_argument('--scene_file', type=str, required=True,
                        help='Path to scene file.')
    parser.add_argument('--reset_every_action', type=int, default=0,
                        choices=[0, 1], help='Reset env after every action.')
    parser.add_argument('--anchor_in_air', type=str2bool, required=True, 
                        default=0, help='Keep anchor in air or not.')
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    main(args)
