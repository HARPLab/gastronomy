import numpy as np
import argparse
import os
import pickle
import json
import pprint
import cv2
import shutil
import logging
import math
import itertools

from sim.sim_vrep.utilities import vrep_utils as vu
from sim.sim_vrep.utilities.math_utils import Sat3D, sample_from_edges
from sim.sim_vrep.utilities.math_utils import get_transformation_matrix_for_vrep_transform_list
from sim.sim_vrep.utilities.math_utils import create_dense_voxels_from_sparse
from sim.sim_vrep.multi_shape_scene_2 import VrepScene

from sim.sim_vrep.lib import vrep

from utils.colors import bcolors

npu = np.random.uniform
npri = np.random.randint

DT = 0.01
DEFAULT_DT = 0.05

def _array_to_str(a, precision=4):
    s = None
    if type(a) is list:
        s = ''
        for l in a:
            s +=  '{:.4f}, '.format(l)
        # Remove last ", "
        s = s[:-2]
    elif type(a) is np.ndarray:
        s = np.array_str(a, precision=precision, suppress_small=True, 
                         max_line_width=120)
    else:
        raise ValueError("Invalid type: {}".format(type(a)))

    return s

class RobotVrepScene(VrepScene):
    def __init__(self, args):
        super(RobotVrepScene, self).__init__(args)

    def set_joint_target_velocity(self, handle, target_velocity):
        response = vrep.simxSetJointTargetVelocity(
            self.client_id,
            handle,
            target_velocity,
            vrep.simx_opmode_oneshot)
        assert response[0] != vrep.simx_return_ok, "cannot set joint target vel"

    def set_joint_force(self, handle, force):
        # This method does not always set the force in the way you expect.
        # It will depend on the control and dynamics mode of the joint.
        response = vrep.simxSetJointForce(
            self.client_id, handle, force, vrep.simx_opmode_oneshot)
        assert response[0] != vrep.simx_return_ok, "cannot set joint target vel"

    def set_all_joint_positions(self, desired_pos):
        assert len(desired_pos) == len(self.handles_dict["joint"])
        # pause simulation
        vrep.simxPauseCommunication(self.client_id, 1)
        for j in range(len(desired_pos)):
            vrep.simxSetJointTargetPosition(
                self.client_id,
                self.handles_dict["joint"][j],
                desired_pos[j],
                vrep.simx_opmode_oneshot)
        # unpause simulation
        vrep.simxPauseCommunication(self.client_id, 0)

    def get_handles(self):
        client_id = self.client_id
        _handle_by_name = lambda n: vu.get_handle_by_name(client_id, n)

        joint_names = [
            "Prismatic_joint_X",
            "Prismatic_joint_Y",
            "Prismatic_joint_Z",
            "Revolute_joint_Z",
            "Revolute_joint_Y",
            "Revolute_joint_X",
            # "Prismatic_joint_Fing1",
            # "Prismatic_joint_Fing2"
        ]
        joint_handles =[_handle_by_name(name) for name in joint_names]
    
        ft_names = ["FT_sensor_EE"]
        ft_handles = [_handle_by_name(n) for n in ft_names]

        octree_handle = _handle_by_name("Octree")
        vision_sensor = _handle_by_name("Vision_sensor")

        robot_handles_dict = dict(
            joint=joint_handles,
            ft=ft_handles,
            octree=octree_handle,
            vision_sensor=vision_sensor,
            # "EE": ee_handles,
            # "FT":ft_handles,
            # "RGB":cam_rgb_handles,
            # "Depth":cam_depth_handles
        )
  
  
        ## Start Data Streaming
        joint_pos =[vu.get_joint_position(client_id, h) for h in joint_handles]
        '''
        EEPose=vrep.simxGetObjectPosition(clientID,RobotHandles["EE"][0],-1,vrep.simx_opmode_streaming)[1];
        EEPose.extend(vrep.simxGetObjectOrientation(clientID,RobotHandles["EE"][0],-1,vrep.simx_opmode_streaming)[1]);      
        FTVal=[vrep.simxReadForceSensor(clientID,FTHandle,vrep.simx_opmode_streaming)[2:4] for FTHandle in RobotHandles["FT"]];
        RobotVisionRGB=[vrep.simxGetVisionSensorImage(clientID,sensorHandle,0,vrep.simx_opmode_streaming) for sensorHandle in RobotHandles["RGB"]];
        RobotVisionDepth=[vrep.simxGetVisionSensorDepthBuffer(clientID,sensorHandle,vrep.simx_opmode_streaming) for sensorHandle in RobotHandles["Depth"]];
        '''
    
        return robot_handles_dict 

    def stop_object_controller(self, obj_handle, dt):
        '''Apply force and torque to stop object motion.'''
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_stopObjectController",
            [obj_handle],
            [dt],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in apply force: {}".format(res[0]))
            return False
        force, torque = res[2][0:3], res[2][3:]
        return force, torque

    def free_other_object(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "RobotBase",
            vrep.sim_scripttype_childscript,
            "ms_freeOtherObject",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in free_other_object: {}".format(res[0]))
            return False
        return True

    def reattach_other_object(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "RobotBase",
            vrep.sim_scripttype_childscript,
            "ms_reAttachOtherObject",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in reattach_other_object: {}".format(res[0]))
            return False
        return True
    
    def get_joint_pid_values(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "RobotBase",
            vrep.sim_scripttype_childscript,
            "ms_GetPIDValues",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in get_joint_pid_values: {}".format(res[0]))
            return None
        return res[2]
    
    def set_joint_pid_values(self, p, i, d):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "RobotBase",
            vrep.sim_scripttype_childscript,
            "ms_SetPIDValues",
            [],
            [p, i, d],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in set_joint_pid_values: {}".format(res[0]))
            return False
        return True


    def get_vision_image(self):
        # vrep.simxReadVisionSensor(
        #     self.client_id,
        #     self.handles_dict['vision_sensor'],
        #     vrep.simx_opmode_blocking)
        resp = vrep.simxGetVisionSensorImage(
            self.client_id,
            self.handles_dict['vision_sensor'],
            0,
            vrep.simx_opmode_blocking,
        )
        status = resp[0]
        img_res = resp[1]
        img_data = np.array(resp[2], dtype=np.uint8).reshape(
            img_res[0], img_res[1], -1)
        img_data = np.flipud(img_data)   # Flip image (to fix it)
        img_data = img_data[:, :, ::-1]  # Convert BGR to RGB
        # import cv2
        # cv2.imwrite('./temp_vision_sensor.png', img_data)
        return img_data

    def get_contacts_info(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetContactsInfo",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in getting contactsInfo: {}".format(res[0]))
            return False

        num_contacts_total = int(res[1][0])
        num_contacts_other = int(res[1][1])
        # parse the contact info
        if num_contacts_other > 0:
            assert len(res[2]) > 0
            info_per_contact = len(res[2]) / num_contacts_other
            all_contact_data = np.array(res[2]).reshape(num_contacts_other, -1)
            logging.info("==== FOUND CONTACT ====: contacts ret: {} "
                         "parsed: {}, info_per_contact: {}".format(
                             num_contacts_total, num_contacts_other,
                             all_contact_data.shape[1]))
        else:
            logging.info(" ===== NO CONTACT =====")
            all_contact_data = None

        return all_contact_data
    
    def get_position_for_anchor_obj(self, obj_size):
        '''Get position to place the anchor object.'''
        # Find a small cuboid within which we can place the anchor object.
        xs, ys, zs = obj_size
        obj_range_x = (-0.25, 0.25)
        obj_range_y = (-0.25, 0.25)
        obj_range_z = (zs/2.0, 0.5)

        pos_x = npu(obj_range_x[0], obj_range_x[1])
        pos_y = npu(obj_range_y[0], obj_range_y[1])
        pos_z = npu(obj_range_z[0], obj_range_z[1])
        return [pos_x, pos_y, pos_z]
    
    def get_anchor_object_args(self, sample_any_pos=False):
        # obj_type = npri(0, 4)
        obj_type = 0
        if obj_type == 1:
            obj_size = [npu(0.02, 0.06)] * 3
        else:
            obj_size = [npu(0.02, 0.12) for i in range(3)]

        # assume the anchor object can be placed in some range
        if sample_any_pos:
            pos = self.get_position_for_anchor_obj()
        else:
            # Create above ground so that it falls down
            pos = [0, 0, obj_size[-1]/2.0]  

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

    def save_scene_data(self, save_dir, a_idx, scene_info,
                        voxels_before_dict,
                        voxels_after_dict,
                        before_contact_info,
                        after_contact_info,
                        recorded_contact_info):
        before_contact_info = before_contact_info.tolist() \
            if before_contact_info is not None else None
        after_contact_info = after_contact_info.tolist() \
            if after_contact_info is not None else None

        self.scene_info[a_idx]['contact'] = {
            'before_contact': before_contact_info,
            'after_contact':  after_contact_info,
        }

        cv2.imwrite(os.path.join(
            save_dir, '{}_before_vision_sensor.png'.format(a_idx)),
            voxels_before_dict['img'])
        if voxels_after_dict is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{}_after_vision_sensor.png'.format(a_idx)),
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

        dense_voxel_data = dict()
        dense_voxel_data['voxels_before'] = dict(
            anchor=create_dense_voxels_from_sparse(
                voxels_before_dict['anchor_voxels'], 
                get_transformation_matrix_for_vrep_transform_list(
                    scene_info[a_idx]['before']['anchor_T_matrix'])
            ),
            other=create_dense_voxels_from_sparse(
                voxels_before_dict['other_voxels'], 
                get_transformation_matrix_for_vrep_transform_list(
                    scene_info[a_idx]['before']['other_T_matrix'])
            )
        )

        if voxels_after_dict is not None:
            dense_voxel_data['voxels_after'] = dict(
                other=create_dense_voxels_from_sparse(
                    voxels_after_dict['other_voxels'], 
                    get_transformation_matrix_for_vrep_transform_list(
                        scene_info[a_idx]['after']['other_T_matrix'])
                )
            )
        
        contact_data = {
            'before_contact': before_contact_info,
            'after_contact': after_contact_info,
            'recorded_contact': recorded_contact_info,
        }

        pkl_path = os.path.join(
            save_dir, '{}_img_data.pkl'.format(a_idx))
        with open(pkl_path, 'wb') as pkl_f:
            pickle.dump(pkl_data, pkl_f, protocol=2)
        pkl_voxel_path = os.path.join(
            save_dir, '{}_voxel_data.pkl'.format(a_idx))
        with open(pkl_voxel_path, 'wb') as pkl_f:
            pickle.dump(voxel_data, pkl_f, protocol=2)
        pkl_dense_voxel_path = os.path.join(
            save_dir, '{}_dense_voxel_data.pkl'.format(a_idx))
        with open(pkl_dense_voxel_path, 'wb') as pkl_f:
            pickle.dump(dense_voxel_data, pkl_f, protocol=2)

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

    def generate_object_with_args(self, obj_args):
        return self.generate_object(
            obj_args['name'],
            obj_args['obj_type'],
            obj_args['obj_size'],
            obj_args['pos'],
            obj_args['q'],
            obj_args['color'],
            obj_args['mass'],
            obj_args['lin_damp'],
            obj_args['ang_damp'],
            obj_args.get('is_static', 0),
        )

    def generate_object(self, name, shape_type, size, pos, q, color, mass,
                        lin_damp, ang_damp, is_static=0):
        '''
        shape_type: Int. 0 for a cuboid, 1 for a sphere, 2 for a cylinder and
            3 for a cone.
        '''
        assert shape_type >= 0 and shape_type <= 3
        assert len(size) == 3, "Invalid size object"
        float_args = [
            # Size        - 0, 1, 2
            size[0], size[1], size[2],
            # pos         - 3, 4, 5
            pos[0], pos[1], pos[2],
            # orientation - 6, 7, 8
            q[0], q[1], q[2],
            # color       - 9, 10, 11
            color[0], color[1], color[2],
            # mass        - 12 mass,
            # linear damp, angular damp     - 13, 14
            lin_damp, ang_damp
        ]

        assert is_static in (0, 1), "Invalid values for is_static"
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_AddBlock",
            [shape_type, is_static],
            float_args,
            [name],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        assert res[0] == vrep.simx_return_ok, "Did not create object"
        obj_handle = res[1][0]
        logging.debug("Created object with handle: {}".format(obj_handle))

        return obj_handle

    def get_other_object_args(self, pos, add_obj_z=True, obj_size=None):
        obj_type = npri(0, 4)
        obj_type = 0
        if obj_size is None:
            obj_size = [npu(0.02, 0.08) for i in range(3)]
        print("Obj size: {}".format(obj_size))
        if add_obj_z:
            pos[-1] = pos[-1] + obj_size[-1]/2.0
        # pos = [-0.4, 0, obj_size[-1]/2.0]
        # q = [npu(-math.pi, math.pi) for i in range(3)]
        # q[1] = npu(-math.pi/2.0, math.pi/2.0)
        q = [0, 0, 0]
        color = [0.0, 1.0, 0.0]
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
        anchor_lb, anchor_ub, other_obj_size, outside_xyz):

        xs, ys, zs = other_obj_size

        pos_list = []
        for i in range(3):
            if outside_xyz[i]:
                pos_i = sample_from_edges(
                    anchor_lb[i]-other_obj_size[i], 
                    anchor_ub[i]+other_obj_size[i], 
                    other_obj_size[i]/2.0 - 0.01,
                    region=None)
            else:
                pos_i = sample_from_edges(
                    anchor_lb[i], 
                    anchor_ub[i], 
                    other_obj_size[i]/2.0 - 0.01)
            
            pos_list.append(pos_i)
    
        return pos_list

    def get_other_obj_location_outside_anchor_cuboid(self, anchor_args, 
        anchor_pos, anchor_lb, anchor_ub, other_obj_size):
        '''Get position to move other object to which is outside anchor cuboid.

        anchor_lb: Absolute lower bound of bounding box.
        anchor_ub: Absolute upper bound of bounding box.

        Return: Position to move other object to.
        '''
        xs, ys, zs = [(other_obj_size[i]+0.01)/2.0 for i in range(3)]

        sample_region_x = (anchor_lb[0]-0.06, anchor_ub[0]+0.06)
        sample_region_y = (anchor_lb[1]-0.06, anchor_ub[1]+0.06)
        sample_region_z = (anchor_lb[2], anchor_ub[2]+ zs + 0.01)

        within_sample_reg_x = (anchor_lb[0]+0.01, anchor_ub[0]-0.01)
        within_sample_reg_y = (anchor_lb[1]+0.01, anchor_ub[1]-0.01)
        within_sample_reg_z = (anchor_ub[2]+zs, anchor_ub[2]+zs+0.02)

        max_tries, idx = 1000, 0
        sample_out_axes = lambda x: npu(0,1) < x
        outside_xyz = [False, False, False] 
        # with 0.6 prob we will sample a point right above the anchor object.
        sample_within_anchor_region = False
        if npu(0, 1) < 0.3:
            outside_xyz = [False, False, True]
            sample_region_x = within_sample_reg_x
            sample_region_y = within_sample_reg_y
            sample_region_z = within_sample_reg_z
            sample_within_anchor_region = True

        while not np.any(outside_xyz):
            outside_xyz = [sample_out_axes(0.5) for _ in range(3)]
        # import ipdb; ipdb.set_trace()

        sample_on_edge = False
        if npu(0, 1) < 0.3:
            sample_on_edge = True
            x, y, z = self.sample_edge_location_for_other_obj(
                anchor_args, anchor_pos, anchor_lb, anchor_ub,
                other_obj_size, outside_xyz)

        info = {'sample_within_anchor_region': sample_within_anchor_region,
                'sample_on_edge': sample_on_edge,
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
            break
            
        if idx >= max_tries:
            return None, info
        
        # x, y = anchor_ub[0]+0.01, anchor_ub[1]+0.01
        
        return [x, y, z], info 

    
    def get_other_obj_position(self, anchor_args, anchor_pos, anchor_lb,
                               anchor_ub, other_obj_size):
        '''Get position of the other object to move it near the main object.

        anchor_lb: Absolute lower bound of bounding box.
        anchor_ub: Absolute upper bound of bounding box.
        '''

        sample_region_x = (anchor_lb[0]-0.1, anchor_ub[0]+0.1)
        sample_region_y = (anchor_lb[1]-0.1, anchor_ub[1]+0.1)
        sample_region_z = (anchor_lb[2], anchor_ub[2]+0.1)
        eye_T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

        max_tries, idx = 1000, 0
        while idx < max_tries:
            x = npu(sample_region_x[0], sample_region_x[1])
            y = npu(sample_region_y[0], sample_region_y[1])
            z = npu(sample_region_z[0], sample_region_z[1])
            idx = idx + 1
            # Let (x, y, z) be the lb of the other object
            lb = [x, y, z]
            ub = [lb[i] + other_obj_size[i] for i in range(3)]
            sat_3d = Sat3D(anchor_lb, anchor_ub, lb, ub, eye_T, eye_T)
            all_dist = sat_3d.get_all_axes_distance()
            
            if idx % 100 == 0:
                print("Did not find other obj position: {}/{}".format(
                    idx, max_tries))

            # Overlap
            if min(all_dist) < 0:
                continue
            break
        
        if idx >= max_tries:
            return None
        
        other_pos = [x+other_obj_size[0]/2.0, 
                     y+other_obj_size[1]/2.0,
                     z+other_obj_size[2]/2.0]

        return other_pos
    
    def get_joint_position(self):
        joint_pos = [vu.get_joint_position(self.client_id, h) 
                     for h in self.handles_dict["joint"]]
        return joint_pos

    def get_all_objects_info(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetAllObjectsInfo",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in apply force on object: {}".format(res[0]))
            return False
        # Parse return values
        assert len(res[2]) == 66, "Got invalid number of return values"
        d = {}
        d['anchor_pos'] = res[2][:3]
        d['anchor_q'] = res[2][3:6]
        d['anchor_v'] = res[2][6:9]
        d['anchor_ang_v'] = res[2][9:12]
        d['other_pos'] = res[2][12:15]
        d['other_q'] = res[2][15:18]
        d['other_v'] = res[2][18:21]
        d['other_ang_v'] = res[2][21:24]
        d['anchor_bb'] = res[2][24:30]
        d['other_bb'] = res[2][30:36]
        d['anchor_T_matrix'] = res[2][36:48]
        d['other_T_matrix'] = res[2][48:60]
        d['joint_pos'] = res[2][60:66]

        return True, d

    def run_scene(self, scene_i, save_dir):
        self.scene_info = {}
        args = self.args

        logging.info("Start sim: {}".format(scene_i))
        self.reset()
        self.step()

        # Generate anchor handle
        anchor_pos = [0, 0, 0.1]
        anchor_args = self.get_anchor_object_args()
        anchor_handle = self.generate_object_with_args(anchor_args)
        self.handles_dict['anchor_obj'] = anchor_handle
        for _ in range(4):
            self.step()
        anchor_pos = self.get_object_position(anchor_handle)
        logging.info("Anchor pos: {}".format(_array_to_str(anchor_pos)))

        # Move the robot to nearby the anchor object.
        orig_joint_position = self.get_joint_position()
        orig_joint_position = [anchor_pos[0], anchor_pos[1]+0.1, 0, 0, 0, 0]
        self.set_all_joint_positions(orig_joint_position)

        # Generate the other object
        other_pos = [0, -0.1, 0.0]
        other_obj_args = self.get_other_object_args(other_pos, add_obj_z=False)
        other_handle = self.generate_object_with_args(other_obj_args)
        self.handles_dict['other_obj'] = other_handle
        for _ in range(5):
            self.step()

        # Move other obj to the right place
        anchor_abs_bb = self.get_absolute_bb(anchor_handle)
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        # other_obj_pos = self.get_other_obj_position(
        #     anchor_args, anchor_pos, anchor_lb, anchor_ub,
        #     other_obj_args['obj_size']
        # )
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
        actions = actions[:1]
        gravity = [0, 1] 
        all_actions = itertools.product(actions, gravity)

        for a_idx, a in enumerate(all_actions):
            # Add action and gravity as one list

            a = a[0] + [a[1]]
            self.scene_info[a_idx] = { 'action': a }
            logging.info(bcolors.c_red(
                " ==== Scene {} action: {} ({}/{}) start ====".format(
                    scene_i, a, a_idx, len(actions))))
            
            if args.reset_every_action == 1 and a_idx > 0:
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

            logging.info(bcolors.c_yellow("Should move other obj to: {}".format(
                _array_to_str(new_other_obj_pos))))
            # Now move the object to near the anchor object
            self.set_all_joint_positions(new_other_obj_pos + [0, 0, 0])
            for _ in range(25):
                self.step()
            joint_pos = self.get_joint_position()
            logging.info(bcolors.c_red(
                "outside_xyz: {}, new joint pos: {}".format(
                sample_info['outside_xyz'], _array_to_str(joint_pos))))

            _, first_obj_data_dict = self.get_all_objects_info()
            self.scene_info[a_idx]['before_dist'] = self.get_object_distance()[-1]

            # Increase the pid values before taking the action.
            old_pid_values = self.get_joint_pid_values()
            self.set_joint_pid_values(0.1, 0.0, 0.0)
            new_pid_values = self.get_joint_pid_values()
            logging.debug("Old pid values: {}\n"
                          "new pid values: {}\n".format(
                            _array_to_str(old_pid_values),
                            _array_to_str(new_pid_values)
                         ))

            # save vision and octree info before taking action
            before_contact_info = self.get_contacts_info()
            voxels_before_dict = self.run_save_voxels_before(
                anchor_handle, other_handle)
            self.toggle_recording_contact_data()
            self.step()

            # Now perform the action
            # after_action_joint_pos = [joint_pos[0], joint_pos[1], anchor_pos[2], 0, 0, 0]
            after_action_joint_pos = [new_other_obj_pos[i] + a[i] 
                                      for i in range(3)]
            self.set_all_joint_positions(after_action_joint_pos + [0, 0, 0])
            for _ in range(20):
                self.step()
            
            # Remove the object from grasp if gravity enabled.
            if a[-1] == 1:
                logging.info(bcolors.c_yellow("Will remove object from grasp"))
                self.free_other_object()
                self.set_joint_pid_values(0.1, 0.0, 0.0)
                for _ in range(15):
                    self.step()

                curr_joint_pos = self.get_joint_position()
                self.reattach_other_object()
                self.set_all_joint_positions(curr_joint_pos)
                for _ in range(20):
                    self.step()

            _, second_obj_data_dict = self.get_all_objects_info()
            obj_in_place = self.are_pos_and_orientation_similar(
                first_obj_data_dict, second_obj_data_dict)
            self.debug_initial_final_position(
                first_obj_data_dict, second_obj_data_dict)
            self.scene_info[a_idx]['0_before'] = first_obj_data_dict 
            self.scene_info[a_idx]['1_after'] = second_obj_data_dict 
            if not obj_in_place: 
                logging.info("Objects changed position AFTER action.")
            self.scene_info[a_idx]['after_dist'] = self.get_object_distance()[-1]

            self.toggle_recording_contact_data()
            _, recorded_contact_info = self.save_recorded_contact_data()
            self.step()

            # Now save vision and octree info after takign action.
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
            )
            before_contact_len = 0 if before_contact_info is None else before_contact_info.shape[0]
            after_contact_len = 0 if after_contact_info is None else after_contact_info.shape[0]
            logging.info("Contact len: before: {}, after: {}".format(
                before_contact_len, after_contact_len))
            logging.info(" ==== Scene {} action: {} Done ====".format(
                scene_i, a))
            if args.reset_every_action == 1:
                self.reset()

        return True


def main(args): 
    scene = RobotVrepScene(args)
    num_scenes, total_try = 0, 0
    while num_scenes < args.num_scenes:
        save_path = os.path.join(args.save_dir, '{:0>5d}'.format(num_scenes))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        status = scene.run_scene(num_scenes, save_path)
        total_try += 1
        if not status:
            shutil.rmtree(save_path)
            print("Incorrect scene, scenes finished: {}, succ: {:.2f}".format(
                num_scenes, float(num_scenes)/total_try))
        else:
            num_scenes += 1
            print("Correct scene: scenes finished: {}, succ: {:.2f}".format(
                num_scenes, float(num_scenes)/total_try))
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
