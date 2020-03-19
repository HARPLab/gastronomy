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

from utilities import vrep_utils as vu
from utilities.math_utils import Sat3D
from utilities.math_utils import get_transformation_matrix_for_vrep_transform_list
from utilities.math_utils import create_dense_voxels_from_sparse
from sim.sim_vrep.multi_shape_scene_2 import VrepScene


from lib import vrep
from utils.colors import bcolors

npu = np.random.uniform
npri = np.random.randint

DT = 0.01
DEFAULT_DT = 0.05

TEST_MOVE_IN_DIR = 0

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
        assert len(res[2]) == 72, "Got invalid number of return values"
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
        d['joint_target_pos'] = res[2][66:72]

        return True, d

    def get_contacts_info_for_other(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetContactsInfoForOther",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in getting contactsInfoForOther: {}".format(res[0]))
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
    
    def set_prismatic_joints(self, desired_pos):
        assert len(desired_pos) == 3, "Invalid number of desired pos."
        vrep.simxPauseCommunication(self.client_id, 1)
        for j in range(len(desired_pos)):
            vrep.simxSetJointTargetPosition(
                self.client_id,
                self.handles_dict["joint"][j],
                desired_pos[j],
                vrep.simx_opmode_oneshot)
        # unpause simulation
        vrep.simxPauseCommunication(self.client_id, 0)

    def set_revolute_joints(self, desired_pos):
        assert len(desired_pos) == 3, "Invalid number of desired pos."
        vrep.simxPauseCommunication(self.client_id, 1)
        for j in range(len(desired_pos)):
            vrep.simxSetJointTargetPosition(
                self.client_id,
                self.handles_dict["joint"][3 + j],
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
        # vision_sensor = _handle_by_name("Vision_sensor")

        robot_handles_dict = dict(
            joint=joint_handles,
            ft=ft_handles,
            octree=octree_handle,
            # vision_sensor=vision_sensor,
            # "EE": ee_handles,
            # "FT":ft_handles,
            # "RGB":cam_rgb_handles,
            # "Depth":cam_depth_handles
        )
  
  
        ## Start Data Streaming
        joint_pos =[vu.get_joint_position(client_id, h) for h in joint_handles]
    
        return robot_handles_dict 

    def get_joint_offsets(self):
        '''Get joint offsets in the scene. 

            Performs joint_pos - joint_target_pos on each joint.
        '''
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "RobotBase",
            vrep.sim_scripttype_childscript,
            "ms_GetJointOffsets",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in  ms_GetJointOffsets: {}".format(
                res[0]))
            return None
        return res[2]
    
    def toggle_record_ft_data(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_toggleRecordingFTData",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in  ms_toggleRecordingFTData: {}".format(
                res[0]))
            return False
        return True

    def save_record_ft_data(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_saveRecordedFTData",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in  ms_saveRecordedFtData: {}".format(res[0]))
            return None

        all_ft_data = res[2]
        return all_ft_data

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
    
    def set_diff_joint_pid_values(self, pid_list):
        assert len(pid_list) == 6
        for i in range(6):
            assert len(pid_list[i]) == 3, "incorrect number of pid values."

        float_inp = []
        for i in range(6):
            for j in range(3):
                float_inp.append(pid_list[i][j])

        res = vrep.simxCallScriptFunction(
            self.client_id,
            "RobotBase",
            vrep.sim_scripttype_childscript,
            "ms_SetAllPIDValues",
            [],
            float_inp,
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in set_diff_joint_pid_values: {}".format(res[0]))
            return False
        return True
        
    def update_octree_position(self, pos):
        assert len(pos) == 3, "Invalid position to be set"
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_UpdateOctreePosition",
            [],
            [pos[0], pos[1], pos[2]],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in update_octree_position: {}".format(res[0]))
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
            "ms_AddAnchorBlock",
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

    def get_anchor_object_args(self):
        # obj_type = npri(0, 4)
        obj_type = 0
        if obj_type == 1:
            obj_size = [npu(0.02, 0.06)] * 3
        else:
            obj_size = [npu(0.06, 0.12) for i in range(3)]

        obj_size = [0.08, 0.08, 0.08]        

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
    
    def get_other_obj_location_outside_anchor_cuboid(self, anchor_args, 
        anchor_pos, anchor_lb, anchor_ub, other_obj_size):
        '''Get position to move other object to which is outside anchor cuboid.

        anchor_lb: Absolute lower bound of bounding box.
        anchor_ub: Absolute upper bound of bounding box.

        Return: Position to move other object to.
        '''
        xs, ys, zs = other_obj_size
        sample_region_x = (anchor_lb[0]-0.0, anchor_ub[0]+0.0)
        sample_region_y = (anchor_lb[1]-0.0, anchor_ub[1]+0.0)
        sample_region_z = (anchor_lb[2], anchor_ub[2]+0.06)

        max_tries, idx = 1000, 0
        sample_out_axes = lambda x: npu(0,1) < x
        # outs ide_xyz = [False]*3
        outside_xyz = [False, False, True] 
        while not np.any(outside_xyz):
            outside_xyz = [sample_out_axes(0.5) for _ in range(3)]
        # import ipdb; ipdb.set_trace()

        while idx < max_tries:
            x = npu(sample_region_x[0], sample_region_x[1])
            y = npu(sample_region_y[0], sample_region_y[1])
            z = npu(sample_region_z[0], sample_region_z[1])
            idx = idx + 1
            if idx % 100 == 0:
                print("Did not find other obj position: {}/{}".format(
                    idx, max_tries))
            if outside_xyz[0] and x >= anchor_lb[0]-0.04 and x <= anchor_ub[0]+0.04:
                continue
            if outside_xyz[1] and y >= anchor_lb[1]-0.04 and y <= anchor_ub[1]+0.04:
                continue
            if outside_xyz[2] and z <= anchor_ub[2]+0.02:
                continue
            break
            
        if idx >= max_tries:
            return None
        
        # HARD-CODE positions for testing
        if TEST_MOVE_IN_DIR == 0:
            x, y = anchor_ub[0]+0.01, anchor_ub[1]+0.01
            # x = (anchor_lb[0]+anchor_ub[0])/2.0
            # y = (anchor_lb[1]+anchor_ub[1])/2.0
        elif TEST_MOVE_IN_DIR == 1:
            x, y, z = anchor_ub[0]+xs/2-0.01, anchor_ub[1]+ys/2.0+0.01, anchor_ub[2]+zs/2.0-0.01
        elif TEST_MOVE_IN_DIR == 2:
            x, y, z = anchor_ub[0]+xs/2+0.01, anchor_ub[1]+ys/2-0.01, anchor_ub[2]+zs/2-0.01
        
        return [x, y, z]

    
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
        '''Get joint position using streaming communication.'''
        joint_pos = [vu.get_joint_position(self.client_id, h) 
                     for h in self.handles_dict["joint"]]
        return joint_pos

    def get_joint_position_oneshot(self):
        '''Get joint position using oneshot communication.'''
        joint_pos = [vu.get_joint_position_oneshot(self.client_id, h) 
                     for h in self.handles_dict["joint"]]
        return joint_pos


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
        for _ in range(4):
            self.step()

        # Generate the other object
        other_pos = [0.0, 0.0, 0.0]
        other_obj_args = self.get_other_object_args(
            other_pos, obj_size=[0.04, 0.04, 0.04], add_obj_z=False)
        other_handle = self.generate_object_with_args(other_obj_args)
        for _ in range(5):
            self.step()

        # Move other obj to the right place
        anchor_abs_bb = self.get_absolute_bb(anchor_handle)
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        # other_obj_pos = self.get_other_obj_position(
        #     anchor_args, anchor_pos, anchor_lb, anchor_ub,
        #     other_obj_args['obj_size']
        # )
        new_other_obj_pos = self.get_other_obj_location_outside_anchor_cuboid(
            anchor_args, 
            anchor_pos,
            anchor_lb,
            anchor_ub,
            other_obj_args['obj_size'])
        if new_other_obj_pos is None:
            logging.info("Cannot place other obj near anchor.")
            return

        print("Should move other obj to: {}".format(
            _array_to_str(new_other_obj_pos)))

        anchor_pos = self.get_object_position(anchor_handle, -1)
        logging.info("Anchor pos: {}".format(_array_to_str(anchor_pos)))

        joint_handles = self.handles_dict['joint']
        joint_pos = [vu.get_joint_position(self.client_id, h)
                     for h in self.handles_dict["joint"]]
        logging.info("Current joint pos: {}".format(_array_to_str(joint_pos)))


        self.set_all_joint_positions(new_other_obj_pos + [0, 0, 0])
        for _ in range(10):
            self.step()
        joint_pos = [vu.get_joint_position_oneshot(self.client_id, h)
                     for h in self.handles_dict["joint"]]
        logging.info("New joint pos: {}".format(_array_to_str(joint_pos)))

        old_pid_values = self.get_joint_pid_values()
        # self.set_joint_pid_values(10.0, 0.0, 1.0)
        '''
        self.set_diff_joint_pid_values([
            [0.01, 0, 0], +1.0000e-01
            [0.01, 0, 0], 
            [0.01, 0, 0], 
            [0.05, 0, 0],
            [0.05, 0, 0],
            [0.05, 0, 0],
        ])
        '''
        new_pid_values = self.get_joint_pid_values()
        logging.info("Old pid values: {}\n"
                     "new pid values: {}\n".format(
                         _array_to_str(old_pid_values),
                         _array_to_str(new_pid_values)
                     ))

        # Now try to perform actions
        # Try to go down for now
        if TEST_MOVE_IN_DIR == 0:
            after_action_joint_pos = [joint_pos[0], joint_pos[1], anchor_pos[2]]
        elif TEST_MOVE_IN_DIR == 1:
            after_action_joint_pos = [joint_pos[0], anchor_pos[1], joint_pos[2]]
        elif TEST_MOVE_IN_DIR == 2:
            after_action_joint_pos = [anchor_pos[0], joint_pos[1], joint_pos[2]]
        
        self.toggle_record_ft_data()
        self.step()
        self.set_prismatic_joints(after_action_joint_pos)
        for _ in range(5):
            joint_pos = self.get_joint_position()
            print(_array_to_str(joint_pos))
            self.step()
        
        '''
        self.free_other_object()
        self.set_joint_pid_values(0.1, 0.0, 0.0)
        for _ in range(20):
            self.step()
        curr_joint_pos = self.get_joint_position()
        self.reattach_other_object()
        self.set_all_joint_positions(curr_joint_pos)
        '''

        for i in range(50):
            self.step()
            if i % 5 == 0:
                joint_pos = self.get_joint_position()
                print(_array_to_str(joint_pos))
        ft_data = self.save_record_ft_data()
        assert ft_data is not None
        ft_data_arr = np.array(ft_data).reshape(-1, 6)
        ft_mean = np.mean(ft_data_arr, axis=0)
        logging.info(bcolors.c_red("Mean ft value: {}".format(
            _array_to_str(ft_mean))))

        return True


def main(args):
    scene = RobotVrepScene(args)
    num_scenes, total_try = 0, 0
    while num_scenes < args.num_scenes:
        # save_path = os.path.join(args.save_dir, str(num_scenes))
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        status = scene.run_scene(num_scenes, None)
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
