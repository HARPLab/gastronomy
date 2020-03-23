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

npu = np.random.uniform
npri = np.random.randint


class RobotVrepScene4OrientChange(RobotVrepScene3):
    def __init__(self, args):
        super(RobotVrepScene3, self).__init__(args)

    def sample_edge_location_for_other_obj(self, anchor_args, anchor_pos,
        anchor_lb, anchor_ub, other_obj_size, outside_xyz, 
        sample_axes_reg):
        '''Sample point on one of the edges of the cuboid.
        
        sample_axes_reg: List of len 3 with values 0, 1, 2 in some order, 
            where 0: in, 1: edge, 2:out
        '''

        xs, ys, zs = other_obj_size

        # import ipdb; ipdb.set_trace()

        pos_list = []
        region_xyz = [None, None, 'high']
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
                    0.05,  # threshold 
                    other_obj_size[i]/2.0,
                    region=region_xyz[i])
                all_info['sample_region_xyz'].append(info['sample_region'])
            else:
                raise ValueError("Invalid sample_axes_reg value")
            
            pos_list.append(pos_i)
        
        assert len(pos_list) == 3
        return pos_list, all_info

    def get_other_obj_location_outside_anchor_cuboid(
        self, anchor_args, anchor_pos, anchor_lb, anchor_ub, other_obj_size, 
        min_x=None, min_y=None, min_z=None):
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
        sample_axes_reg = [npri(3) for i in range(2)]
        # Put both outside with some prob.
        if npu(0, 1) < 0.0:
            sample_axes_reg = [2, 2]
        else:
            sample_axes_reg = [0, 1]
            np.random.shuffle(sample_axes_reg)
            while not np.any(np.int32(sample_axes_reg) == 2):
                sample_axes_reg = [npri(3) for i in range(2)]
        
        # In on the z axes. Set this only for the ground data.
        sample_axes_reg.append(0)

        outside_xyz = []
        for ax in sample_axes_reg:
            outside_val = True if ax == 2 else False
            outside_xyz.append(outside_val)

        sample_on_edge = True
        [x, y, z], sample_edge_info = self.sample_edge_location_for_other_obj(
            anchor_args, anchor_pos, anchor_lb, anchor_ub,
            other_obj_size, outside_xyz, 
            sample_axes_reg,
            )

        # If sampling on ground then make z directly above ground?
        sample_on_ground = True
        if sample_on_ground:
            z = other_obj_size[2]/2.0 
            # Move it slightly above to account for the fact that there can be
            # inaccuracies in real world voxels.
            if npu(0, 1) < 0.3:
                z += npu(0, 0.02)

        info = {'sample_on_edge': False,
                'sample_axes_reg': sample_axes_reg,
                'sample_edge_info': sample_edge_info,
                'outside_xyz': outside_xyz,
                'sample_on_ground': True}

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

        logging.info("Start sim: {}".format(scene_i))
        self.reset()
        self.step()

        # Generate anchor handle
        anchor_pos = [0.0, 0.0, 0.0]
        anchor_args = self.get_anchor_object_args()
        # Move it slightly above to account for the fact that there can be
        # inaccuracies in real world voxels.
        if npu(0, 1) < 0.3:
            anchor_args['pos'][2] += npu(0, 0.02)

        anchor_handle = self.generate_object_with_args(anchor_args)
        self.handles_dict['anchor_obj'] = anchor_handle
        for _ in range(4):
            self.step()
        anchor_pos = self.get_object_position(anchor_handle)
        logging.info("Anchor pos: {}".format(_array_to_str(anchor_pos)))

        # Move the robot to nearby (above) the anchor object.
        orig_joint_position = self.get_joint_position()
        orig_joint_position = [anchor_pos[0], anchor_pos[1], anchor_pos[2]+0.2, 0, 0, 0]
        self.set_all_joint_positions(orig_joint_position)

        # Generate the other object
        other_pos = [0.0, 0.0, 0.0]
        other_obj_args = self.get_other_object_args(other_pos, add_obj_z=False)
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
            other_obj_args['obj_size'])
        if new_other_obj_pos is None:
            logging.info("Cannot place other obj near anchor.")
            return False

        self.scene_info['new_other_obj_pos'] = new_other_obj_pos
        self.scene_info['sample_info'] = sample_info

        actions = [[-.1, 0, 0], [.1, 0, 0], [0, -.1, 0], [0, .1, 0], 
                   [0, 0, -.1], [0, 0, .1]]
        filter_actions_to_max_torque = True
        # import ipdb; ipdb.set_trace()
        if filter_actions_to_max_torque and sample_info['sample_on_edge']:
            axes_to_action_idx = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
            valid_actions = copy.deepcopy(actions)
            for axes_i, axes_out in enumerate(sample_info['outside_xyz']):
                if not axes_out:
                    assert len(axes_to_action_idx[axes_i]) == 2
                    valid_actions[axes_to_action_idx[axes_i][0]] = None
                    valid_actions[axes_to_action_idx[axes_i][1]] = None
                else:
                    region = sample_info['sample_edge_info']['sample_region_xyz'][axes_i]
                    if region == 'low':
                        valid_actions[axes_to_action_idx[axes_i][0]] = None
                    elif region == 'high':
                        valid_actions[axes_to_action_idx[axes_i][1]] = None
                    elif region == 'middle':
                        raise ValueError("Invalid region: {}".format(region))
                    else:
                        raise ValueError("Invalid region: {}".format(region))
            
            logging.info("Filtered actions from {} to {}".format(
                len(actions), len([va for va in valid_actions if va is not None])))
            actions = valid_actions
        elif sample_info['sample_on_ground']:
            actions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]            
        else:
            raise ValueError("WTF")

        # actions = [[-.1, 0, 0], [.1, 0, 0]]
        has_created_scene = True
        for a_idx, a in enumerate(actions):
            if a is None:
                continue

            self.scene_info[a_idx] = { 'action': a }
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
            else:
                # First go to rest position
                self.set_all_joint_positions(orig_joint_position)
                for _ in range(50):
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
            for _ in range(25):
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
            for _ in range(25):
                self.step()

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
                # import ipdb; ipdb.set_trace()
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
                return False
            # if not self.are_objects_close(second_obj_data_dict):
            #     logging.error("Objects are not close. Will sample again.")
            #     return False

            self.scene_info[a_idx]['before_dist'] = self.get_object_distance()[-1]

            # save vision and octree info before taking action
            before_contact_info = self.get_contacts_info()
            # before_ft_data = self.read_force_torque_data()
            voxels_before_dict = self.run_save_voxels_before(
                anchor_handle, other_handle)
            self.toggle_recording_contact_data()
            self.toggle_record_ft_data()
            self.step()

            # Now perform the action
            # after_action_joint_pos = [joint_pos[0], joint_pos[1], anchor_pos[2], 0, 0, 0]
            if sample_info['sample_on_ground']:
                diff = [a[i]*(anchor_args['pos'][i]-new_other_obj_waypoints[-1][i]) 
                        for i in range(2)]
                diff.append(0)
                after_action_joint_pos = [new_other_obj_waypoints[-1][i]+diff[i]
                                          for i in range(3)]
                logging.info(f"After action pos: {after_action_joint_pos}")
            else:
                after_action_joint_pos = [new_other_obj_waypoints[-1][i] + a[i] 
                                        for i in range(3)]
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
            self.debug_initial_final_position(
                second_obj_data_dict, third_obj_data_dict)
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
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    main(args)
