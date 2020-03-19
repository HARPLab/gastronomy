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
from sim.sim_vrep.robot_scene_1 import RobotVrepScene
from sim.sim_vrep.robot_scene_1 import _array_to_str

from lib import vrep
from utils.colors import bcolors

npu = np.random.uniform
npri = np.random.randint


class RobotVrepScene3(RobotVrepScene):
    def __init__(self, args):
        super(RobotVrepScene3, self).__init__(args)

    def save_scene_data(self, save_dir, a_idx, scene_info,
                        voxels_before_dict,
                        voxels_after_dict,
                        before_contact_info,
                        after_contact_info,
                        recorded_contact_info,
                        recorded_ft_data):
        before_contact_info = before_contact_info.tolist() \
            if before_contact_info is not None else None
        after_contact_info = after_contact_info.tolist() \
            if after_contact_info is not None else None

        self.scene_info[a_idx]['contact'] = {
            'before_contact': before_contact_info,
            'after_contact':  after_contact_info,
        }

        cv2.imwrite(os.path.join(
            save_dir, '{}_0_before_vision_sensor.png'.format(a_idx)),
            voxels_before_dict['img'])
        if voxels_after_dict is not None:
            if voxels_after_dict.get('img') is not None:
                cv2.imwrite(os.path.join(
                    save_dir, '{}_1_after_vision_sensor.png'.format(a_idx)),
                    voxels_after_dict['img'])
        pkl_data = {'data': scene_info[a_idx]}
        voxel_data = {
            'voxels_before': {
                'anchor': voxels_before_dict['anchor_voxels'],
                'other': voxels_before_dict['other_voxels']
            },
        }
        if voxels_after_dict is not None:
            voxel_data.update({'voxels_after': {
                    'other': voxels_after_dict.get('other_voxels'),
                }})
        
        contact_data = {
            'before_contact': before_contact_info,
            'after_contact': after_contact_info,
            'recorded_contact': recorded_contact_info,
            'recorded_ft_data': recorded_ft_data,
        }

        pkl_path = os.path.join(
            save_dir, '{}_img_data.pkl'.format(a_idx))
        with open(pkl_path, 'wb') as pkl_f:
            pickle.dump(pkl_data, pkl_f, protocol=2)
        pkl_voxel_path = os.path.join(
            save_dir, '{}_voxel_data.pkl'.format(a_idx))
        with open(pkl_voxel_path, 'wb') as pkl_f:
            pickle.dump(voxel_data, pkl_f, protocol=2)
        pkl_contact_path = os.path.join(
            save_dir, '{}_contact_data.pkl'.format(a_idx))
        with open(pkl_contact_path, 'wb') as pkl_f: 
            pickle.dump(contact_data, pkl_f, protocol=2)

        json_path = os.path.join(
            save_dir, '{}_img_data.json'.format(a_idx))
        with open(json_path, 'w') as json_f:
            json_f.write(json.dumps(scene_info[a_idx]))
        json_path = os.path.join(
            save_dir, 'all_img_data.json'.format(a_idx))
        with open(json_path, 'w') as json_f:
            json_f.write(json.dumps(scene_info))

    
    def get_anchor_object_args(self, min_obj_size=0.02, max_obj_size=0.10):
        # obj_type = npri(0, 4)
        obj_type = 1
        if obj_type == 1:
            obj_size = [npu(min_obj_size, max_obj_size)] * 3
        else:
            obj_size = [npu(min_obj_size, max_obj_size) for i in range(3)]
            obj_size = [0.08, 0.06, 0.06]

        pos = [0, 0, obj_size[-1]/2.0]  # Create above ground so that it falls down
        # q = [npu(-math.pi, math.pi) for i in range(3)]
        q = [0, 0, 0]
        color = [1.0, 1.0, 0]
        mass = 1.0  # Heavy object

        return dict(
            name="anchor",
            obj_type=obj_type,
            obj_size=obj_size,
            pos=pos,
            q=q,
            color=color,
            mass=mass,
            is_static=1,
            lin_damp=0.1,
            ang_damp=0.1,
        )

    def debug_initial_final_position(self, i_dict, f_dict):
        i_dict_arr = dict({k: np.array(v) for k, v in i_dict.items()})
        f_dict_arr = dict({k: np.array(v) for k, v in f_dict.items()})
        logging.info(
            "Vel info\n"
            "           anchor obj   \t   vel: {}, ang_vel: {}\n"
            "            other obj   \t   vel: {}, ang_vel: {}\n".format(
                f_dict_arr['anchor_v'], f_dict_arr['anchor_ang_v'],
                f_dict_arr['other_v'], f_dict_arr['other_ang_v'],
            ))
        logging.info(
            "Pos diff\n"
            "           anchor:    \t   pos: {}, q: {}\n"
            "            other:    \t   pos: {}, q: {}\n".format(
                np.abs(f_dict_arr['anchor_pos'] - i_dict_arr['anchor_pos']),
                np.abs(f_dict_arr['anchor_q'] - i_dict_arr['anchor_q']),
                np.abs(f_dict_arr['other_pos'] - i_dict_arr['other_pos']),
                np.abs(f_dict_arr['other_q'] - i_dict_arr['other_q']),
            ))

    def get_other_object_args(self, pos, add_obj_z=True, obj_size=None,
                              min_size=0.02, max_size=0.10):
        obj_type = npri(0, 4)
        obj_type = 2
        if obj_size is None:
            obj_size = [npu(min_size, max_size) for i in range(3)]
        # obj_size = [0.04, 0.036, 0.024]
        print("Obj size: {}".format(obj_size))
        if add_obj_z:
            pos[-1] = pos[-1] + obj_size[-1]/2.0
        # pos = [-0.4, 0, obj_size[-1]/2.0]
        # q = [npu(-math.pi, math.pi) for i in range(3)]
        # q[1] = npu(-math.pi/2.0, math.pi/2.0)
        q = [0, 0, 0]
        color = [0.0, 1.0, 1.0]
        mass = 0.01  # Heavy object

        return dict(
            name="other",
            obj_type=obj_type,
            obj_size=obj_size,
            pos=pos,
            q=q,
            color=color,
            mass=mass,
            is_static=0,
            lin_damp=0.1,
            ang_damp=0.1,
        )

    def sample_edge_location_for_other_obj(self, anchor_args, anchor_pos,
        anchor_lb, anchor_ub, other_obj_size, outside_xyz, 
        sample_inside_around_edges=True):
        '''Sample point on one of the edges of the cuboid.'''

        xs, ys, zs = other_obj_size

        import ipdb; ipdb.set_trace()

        pos_list = []
        for i in range(2):
            if outside_xyz[i]:
                pos_i, info = sample_from_edges(
                    anchor_lb[i]-other_obj_size[i], 
                    anchor_ub[i]+other_obj_size[i], 
                    other_obj_size[i]/2.0 - 0.01,
                    region=None)
            else:
                if sample_inside_around_edges:
                    pos_i, info = sample_from_edges(
                        anchor_lb[i], 
                        anchor_ub[i], 
                        other_obj_size[i]/2.0 - 0.01)
                else:
                    assert anchor_lb[i]+0.01 < anchor_ub[i]-0.01
                    pos_i = npu(anchor_lb[i] + 0.01, anchor_ub[i] - 0.01)
                    info = {'region': 'middle'}
            
            pos_list.append(pos_i)
        
        # Do this separately for z-axis
        if outside_xyz[2]:
            pos_i, info = sample_from_edges(
                anchor_lb[2], 
                anchor_ub[2]+other_obj_size[2], 
                other_obj_size[2]/2.0 - 0.01,
                region='outside_high')
        else:
            if sample_inside_around_edges:
                pos_i, info = sample_from_edges(
                    anchor_lb[2]+other_obj_size[2], 
                    anchor_ub[2], 
                    other_obj_size[i]/2.0 - 0.01)
            else:
                assert anchor_lb[2]+0.01 < anchor_ub[2]-0.01
                pos_i = npu(anchor_lb[2]+other_obj_size[2]/2.0, anchor_ub[2]-0.01)
                info = {'region': 'middle'}
        pos_list.append(pos_i)

        assert len(pos_list) == 3
        return pos_list, info

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

        sample_region_x = (anchor_lb[0]-0.06, anchor_ub[0]+0.06)
        sample_region_y = (anchor_lb[1]-0.06, anchor_ub[1]+0.06)
        sample_region_z = (anchor_lb[2], anchor_ub[2]+0.06)

        within_sample_reg_x = (anchor_lb[0]+0.01, anchor_ub[0]-0.01)
        within_sample_reg_y = (anchor_lb[1]+0.01, anchor_ub[1]-0.01)
        within_sample_reg_z = (anchor_ub[2]+zs, anchor_ub[2]+zs+0.02)

        max_tries, idx = 1000, 0
        sample_out_axes = lambda x: npu(0,1) < x
        outside_xyz = [False, False, False] 
        # with 0.6 prob we will sample a point right above the anchor object.
        sample_within_anchor_region = False
        if npu(0, 1) < 0.0:
            outside_xyz = [False, False, True]
            sample_region_x = within_sample_reg_x
            sample_region_y = within_sample_reg_y
            sample_region_z = within_sample_reg_z
            sample_within_anchor_region = True
        
        only_one_axes_out = True

        outside_xyz = [False, False, False]
        if only_one_axes_out:
            outside_xyz[npri(0, 3)] = True
        else:
            while not np.any(outside_xyz):
                outside_xyz = [sample_out_axes(0.5) for _ in range(3)]

        sample_on_edge = False
        sample_inside_around_edges = False
        if npu(0, 1) < 1.0:
            sample_on_edge = True
            sample_inside_around_edges = True if npu(0, 1) < 0 else False
            [x, y, z], sample_inside_around_edges_info = self.sample_edge_location_for_other_obj(
                anchor_args, anchor_pos, anchor_lb, anchor_ub,
                other_obj_size, outside_xyz, 
                sample_inside_around_edges=sample_inside_around_edges
                )

        info = {'sample_within_anchor_region': sample_within_anchor_region,
                'sample_on_edge': sample_on_edge,
                'sample_inside_around_edges': sample_inside_around_edges,
                'sample_inside_around_edges_info': sample_inside_around_edges_info,
                'outside_xyz': outside_xyz}

        if sample_on_edge:
            return [x, y, z], info 

        while idx < max_tries:
            x = npu(sample_region_x[0], sample_region_x[1])
            y = npu(sample_region_y[0], sample_region_y[1])
            z = npu(sample_region_z[0], sample_region_z[1])
            idx = idx + 1
            if idx % 100 == 0:
                print("Did not find other obj position: {}/{}".format(
                    idx, max_tries))
            if outside_xyz[0] and x >= anchor_lb[0]-xs and x <= anchor_ub[0]+xs:
                continue
            if outside_xyz[1] and y >= anchor_lb[1]-ys and y <= anchor_ub[1]+ys:
                continue
            if outside_xyz[2] and z <= anchor_ub[2]+zs:
                continue
            if min_z is not None and  z - other_obj_size[2]/2.0 < 0.0:
                continue
            break
            
        if idx >= max_tries:
            return None, info
        
        # x, y = anchor_ub[0]+0.01, anchor_ub[1]+0.01
        
        # return [anchor_pos[0], anchor_pos[1], anchor_pos[2]+0.2], info
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
            return
        self.scene_info['new_other_obj_pos'] = new_other_obj_pos
        self.scene_info['sample_info'] = sample_info

        actions = [[-.1, 0, 0], [.1, 0, 0], [0, -.1, 0], [0, .1, 0], 
                   [0, 0, -.1], [0, 0, .1]]
        filter_actions_to_max_torque = True
        # import ipdb; ipdb.set_trace()
        if filter_actions_to_max_torque and sample_info['sample_on_edge']:
            axes_to_action_idx = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
            valid_actions = copy.deepcopy(actions)
            for axes_i in range(len(sample_info['outside_xyz'])):
                if not axes_i:
                    assert len(axes_to_action_idx[axes_i]) == 2
                    valid_actions[axes_to_action_idx[axes_i][0]] = None
                    valid_actions[axes_to_action_idx[axes_i][1]] = None
                else:
                    region = sample_info['sample_inside_around_edges_info']['region']
                    if region == 'low':
                        valid_actions[axes_to_action_idx[axes_i][0]] = None
                    elif region == 'high':
                        valid_actions[axes_to_action_idx[axes_i][1]] = None
                    elif region == 'middle':
                        pass
                    else:
                        raise ValueError("Invalid region: {}".format(region))
            
            logging.info("Filtered actions from {} to {}".format(
                len(actions), len(valid_actions)))
            actions = valid_actions
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
            logging.info("Mean force-torque val: {}".format(
                _array_to_str(ft_sensor_mean)))

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
    scene = RobotVrepScene3(args)
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
                  "  \t  torque_above_th: {:.2f}".format(
                      num_scenes, float(num_scenes)/total_try,
                      float(num_scenes_above_torque_th)/total_scenes_actions))
        else:
            num_scenes += 1
            if total_scenes_actions > 0:
                torque_above_th_ratio = float(num_scenes_above_torque_th)/total_scenes_actions
            else:
                torque_above_th_ratio = 0.0
            print("Correct scene: scenes finished: {}, succ: {:.2f}"
                  "  \t  torque_above_th: {:.2f}".format(
                      num_scenes, float(num_scenes)/total_try,
                      torque_above_th_ratio))
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
