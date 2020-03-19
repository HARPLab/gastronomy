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

from lib import vrep

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

class VrepScene(object):
    def __init__(self, args):
        self.args = args
        # just in case, close all opened connections
        vrep.simxFinish(-1)
        self.client_id = self.connect(args.scene_file, args.port)
        # set the dt (simulation timestep, default is 0.05)
        self.set_simulation_timestep()
        self.handles_dict = self.get_handles()
        self.set_simulation_timestep()

    def set_simulation_timestep(self):
        ret = vrep.simxSetFloatingParameter(
            self.client_id,
            vrep.sim_floatparam_simulation_time_step,
            DT,
            vrep.simx_opmode_blocking)
        assert ret == vrep.simx_return_ok, "Cannot set sim timestep"

    def stop(self):
        vu.stop_sim(self.client_id)

    def stop_and_finish(self):
        self.stop()
        vrep.simxFinish(self.client_id)

    def reset(self):
        self.stop()
        self.step()
        self.set_simulation_timestep()
        self.step()
        self.start()


    def connect(self, scene_file, port):
        client_id = vu.connect_to_vrep(port=port)
        res = vu.load_scene(client_id, scene_file)
        return client_id

    def start(self):
        vu.start_sim(self.client_id)

    def step(self):
        vu.step_sim(self.client_id)

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

        '''
        ee_names = ["EndEffector"]
        ee_handles = [_handle_by_name(n) for n in ee_names]
  
        ft_names = ["FT_sensor_Wrist", "FT_sensor_EE"]
        ft_handles = [_handle_by_name(n) for n in ft_names]

        cam_rgb_names = ["PalmCam_RGB"]
        cam_rgb_handles =[_handle_by_name(n) for n in cam_rgb_names]
  
        cam_depth_names = ["PalmCam_Depth"]  
        cam_depth_handles =[_handle_by_name(n) for n in cam_depth_names]    
        '''
  
        robot_handles_dict = {
            "joint": joint_handles,
            "ft": ft_handles,
            # "EE": ee_handles,
            # "FT":ft_handles,
            # "RGB":cam_rgb_handles,
            # "Depth":cam_depth_handles
            }
  
  
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


    def get_object_distance(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetObjectDistance",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in get object distance: {}".format(res[0]))
        return res[2]

    def get_absolute_bb(self, obj_handle):
        '''Get the absolute bounding box of an axis-aligned object.
        '''
        anchor_bb = self.get_object_bb(obj_handle)
        anchor_pos = self.get_object_position(obj_handle)
        lb = anchor_bb[0] + anchor_pos
        ub = anchor_bb[1] + anchor_pos
        return lb, ub

    def get_object_bb(self, obj_handle):
        value_list = []
        for param in [15, 16, 17, 18, 19, 20]:
            resp, value = vrep.simxGetObjectFloatParameter(
                self.client_id, obj_handle, param,
                vrep.simx_opmode_blocking)
            value_list.append(value)

        return np.array(value_list[0:3]), np.array(value_list[3:])

    def set_object_position(self, obj_handle, position):
        resp = vrep.simxSetObjectPosition(
            self.client_id,
            obj_handle,
            -1,  # Sets absolute position
            position,
            vrep.simx_opmode_oneshot)
        if resp != vrep.simx_return_ok:
            logging.error("Error in setting object position: {}".format(resp))

    def set_object_orientation(self, obj_handle, euler_angles):
        resp = vrep.simxSetObjectOrientation(
            self.client_id,
            obj_handle,
            -1,  # Sets absolute orientation
            euler_angles,
            vrep.simx_opmode_oneshot)
        if resp != vrep.simx_return_ok:
            logging.error("Error in setting object orientation: {}".format(resp))

    def get_object_velocity(self, handle, streaming=True):
        mode = vrep.simx_opmode_streaming if streaming else \
            vrep.simx_opmode_blocking
        resp, lin_vel, ang_vel = vrep.simxGetObjectVelocity(
            self.client_id, handle, mode)
        if resp != vrep.simx_return_ok:
            logging.warning("Error in getting object velocity")
            return [], []
        return lin_vel, ang_vel

    def get_object_position(self, handle, relative_to_handle=-1):
        '''Return the object position in reference to the relative handle.'''
        response, position = vrep.simxGetObjectPosition(
            self.client_id, handle, relative_to_handle,
            vrep.simx_opmode_blocking)
        if response != vrep.simx_return_ok:
            logging.warning(
                "Error: Cannot query position for handle {} "
                "with reference to {}".format(handle, relative_to_handle))
        return np.array(position)

    def get_object_orientation(self, handle, reference_handle=-1):
        '''Return the object orientation in reference to the relative handle.'''
        response, orientation = vrep.simxGetObjectOrientation(
            self.client_id, handle, reference_handle,
            vrep.simx_opmode_blocking)
        if response != 0:
            logging.warning(
                "Error: Cannot query position for handle {} with reference to {}".
                format(handle, reference_handle))
        return np.array(orientation)

    def get_octree_voxels(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_GetOctreeVoxels",
            [int(self.handles_dict['octree'])],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in getting octree voxels: {}".format(res[0]))
        assert len(res[2]) > 0, "Empty octree voxels"
        return res[2]

    def update_octree(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_UpdateOctreeVoxels",
            [int(self.handles_dict['octree']),
             int(self.handles_dict['anchor_obj']),
             int(self.handles_dict['other_obj'])],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in udpate octree: {}".format(res[0]))
    
    def update_octree_with_single_object(self, obj_handle):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_UpdateOctreeWithSingleObject",
            [obj_handle],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in udpate octree: {}".format(res[0]))

    def clear_octree(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_ClearOctree",
            [int(self.handles_dict['octree'])],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in clear octree: {}".format(res[0]))

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

    def get_object_damping_param(self, obj_handle):
        '''Get object damping parameters.'''
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetEngineParam",
            [obj_handle],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in getting damping param: {}".format(res[0]))
            return False
        lin_damp, ang_damp = res[2][0], res[2][1]
        return lin_damp, ang_damp

    def set_object_damping_param(self, obj_handle, lin_damp, ang_damp):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_SetEngineParam",
            [o_handle],
            [lin_damp, ang_damp],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in setting damping param: {}".format(res[0]))
            return False
        return True

    def toggle_keep_object_in_place(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_ToggleObjectInPlace",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in toggle_keep_in_place: {}".format(res[0]))
            return False
        return True
    
    def toggle_recording_contact_data(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_toggleRecordingContactData",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in  ms_toggleRecordingContactData: {}".format(
                res[0]))
            return False
        return True
        # ms_saveRecordedContactData

    def save_recorded_contact_data(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_saveRecordedContactData",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in  ms_saveRecordedContactData: {}".format(
                res[0]))
            return False

        all_contact_data = res[2]
        return True, all_contact_data

    def update_keep_object_in_place(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_KeepObjectInPlace",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in update_keep_in_place: {}".format(res[0]))
            return False
        return True
    
    def remove_anchor_object(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_RemoveAnchorObject",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in remove_anchor_object: {}".format(res[0]))
            return False
        return True

    def remove_other_object(self):
        # ms_RemoveOtherObject
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_RemoveOtherObject",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in remove_other_object: {}".format(res[0]))
            return False
        return True

    def update_add_object(self, anchor_or_other_obj):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_DidAddObject",
            [anchor_or_other_obj],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in updateAddObject: {}".format(res[0]))
            return False
        return True

    def apply_force(self, o_handle, position, force):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_ApplyForce",
            [o_handle],
            [position[0], position[1], position[2],
             force[0], force[1], force[2]],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in apply force: {}".format(res[0]))
            return False
        return True

    def apply_force_in_ref_frame(self, force):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_ApplyForceInWorldFrameAtCOM",
            [],
            [force[0], force[1], force[2]],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in apply_force_in_ref_frame: {}".format(res[0]))
            return False
        return True

    def apply_force_on_object(self, o_handle, force):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_ApplyForceOnObject",
            [o_handle],
            [force[0], force[1], force[2]],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in apply force on object: {}".format(res[0]))
            return False
        return True
    
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

    def get_oriented_bounding_box(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetOrientedAbsoluteBoundingBox",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in getting oriented bounding box: {}".format(
                res[0]))
            return False

        assert len(res[2]) == 6, "Got invalid number of return values"
        lb, ub = np.array(res[2][:3]), np.array(res[2][3:])
        return lb, ub

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
    
    def get_other_objects_info(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetOtherObjectInfo",
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
        assert len(res[2]) == 30, "Got invalid number of return values"
        d = {}
        d['other_pos'] = res[2][:3]
        d['other_q'] = res[2][3:6]
        d['other_v'] = res[2][6:9]
        d['other_ang_v'] = res[2][9:12]
        d['other_bb'] = res[2][12:18]
        d['other_T_matrix'] = res[2][18:30]

        return True, d

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
        assert len(res[2]) == 60, "Got invalid number of return values"
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

        return True, d

    def get_obj_data_dict(self, obj_handle, obj_prefix):
        d = {}
        obj_bb = self.get_object_bb(obj_handle)
        d[obj_prefix+'_bb'] = obj_bb[0].tolist() + obj_bb[1].tolist()
        d[obj_prefix+'_pos'] = self.get_object_position(obj_handle).tolist()
        d[obj_prefix+'_q'] = self.get_object_orientation(obj_handle).tolist()
        return d

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

    def are_pos_and_orientation_similar(self, i_dict, f_dict, 
                                        verify_other_obj_only=False):
        if not verify_other_obj_only:
            anchor_pos_i = np.array([
                i_dict['anchor_pos'], i_dict['anchor_q']]).reshape(-1)
            anchor_pos_f = np.array([
                f_dict['anchor_pos'], f_dict['anchor_q']]).reshape(-1)
            if np.any(np.abs(anchor_pos_i - anchor_pos_f) > 1e-3):
                return False
        other_pos_i = np.array([
            i_dict['other_pos'], i_dict['other_q']]).reshape(-1)
        other_pos_f = np.array([
            f_dict['other_pos'], f_dict['other_q']]).reshape(-1)
        if np.any(np.abs(other_pos_i - other_pos_f) > 1e-3):
            return False
        return True

    def are_objects_close_old(self, f_dict):
        '''Check if objects are close or not.'''
        # This is not a great test.
        d = np.array(f_dict['anchor_pos']) - np.array(f_dict['other_pos'])
        dist = np.linalg.norm(d, 2)

        # Use SAT test
        sat_3d = Sat3D(
            f_dict['anchor_bb'][:3], f_dict['anchor_bb'][3:],
            f_dict['other_bb'][:3],  f_dict['other_bb'][3:],
            f_dict['anchor_T_matrix'],
            f_dict['other_T_matrix'])
        all_dist = sat_3d.get_all_axes_distance()

        if dist > 0.8:
            return False

        if max(all_dist) > 0.2:
            print("SAT dist: {:.4f}".format(max(all_dist)))
            return False

        return True

    def are_objects_close(self, f_dict):
        '''Check if objects are close or not.'''
        dist = self.get_object_distance()
        return dist[-1] < 0.1

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

    def run_save_voxels_before(self, anchor_handle, other_handle):
        # Get image and octree data
        img_data_before = self.get_vision_image()
        self.update_octree_with_single_object(anchor_handle)
        for _ in range(10):
            self.step()
        anchor_voxels = self.get_octree_voxels()
        logging.debug("Got anchor voxels: {}".format(len(anchor_voxels)))
        self.update_octree_with_single_object(other_handle)
        for _ in range(10):
            self.step()
        other_voxels = self.get_octree_voxels()
        logging.debug("Got anchor voxels: {}".format(len(other_voxels)))
        for _ in range(5):
            self.step()
        self.clear_octree()
        self.step()
        return {
            'img': img_data_before,
            'anchor_voxels': anchor_voxels,
            'other_voxels': other_voxels,
        }
    
    def run_save_voxels_after(self, anchor_handle, other_handle):
        img_data_after = self.get_vision_image()
        self.update_octree_with_single_object(other_handle)
        for _ in range(5):
            self.step()
        other_voxels = self.get_octree_voxels()
        for _ in range(5):
            self.step()
        self.clear_octree()
        self.step()
        return {
            'img': img_data_after,
            'other_voxels': other_voxels,
        }

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
        sample_region_x = (anchor_lb[0]-0.0, anchor_ub[0]+0.0)
        sample_region_y = (anchor_lb[1]-0.0, anchor_ub[1]+0.0)
        sample_region_z = (anchor_lb[2], anchor_ub[2]+0.06)

        max_tries, idx = 1000, 0
        sample_out_axes = lambda x: npu(0,1) < x
        # outside_xyz = [False]*3
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
        
        x, y = anchor_ub[0]+0.01, anchor_ub[1]+0.01
        
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
        joint_pos = [vu.get_joint_position(self.client_id, h) 
                     for h in self.handles_dict["joint"]]
        return joint_pos

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
        for _ in range(4):
            self.step()

        # Generate the other object
        other_pos = [0, -0.1, 0.0]
        other_obj_args = self.get_other_object_args(other_pos, add_obj_z=False)
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
        joint_pos = [vu.get_joint_position(self.client_id, h)
                     for h in self.handles_dict["joint"]]
        logging.info("New joint pos: {}".format(_array_to_str(joint_pos)))

        old_pid_values = self.get_joint_pid_values()
        self.set_joint_pid_values(10.0, 0.0, 1.0)
        new_pid_values = self.get_joint_pid_values()
        logging.info("Old pid values: {}\n"
                     "new pid values: {}\n".format(
                         _array_to_str(old_pid_values),
                         _array_to_str(new_pid_values)
                     ))

        # Now try to perform actions
        # Try to go down for now
        after_action_joint_pos = [joint_pos[0], joint_pos[1], anchor_pos[2], 0, 0, 0]
        self.set_all_joint_positions(after_action_joint_pos)
        for _ in range(5):
            self.step()
        
        self.free_other_object()
        self.set_joint_pid_values(0.1, 0.0, 0.0)
        for _ in range(20):
            self.step()

        curr_joint_pos = self.get_joint_position()
        self.reattach_other_object()
        self.set_all_joint_positions(curr_joint_pos)

        '''
        self.step()
        joint_pos[2] += 0.2
        self.set_all_joint_positions(joint_pos)

        other_pos = self.get_object_position(other_handle, -1)
        logging.info("Other pos: ({:.4f}, {:.4f}, {:.4f})".format(
            other_pos[0], other_pos[1], other_pos[2]))

        # Move to slightly above the anchor block.
        offset = np.array([0.0, 0.0, 0.00])
        # above_anchor_pos = anchor_pos - other_pos + offset
        above_anchor_pos = anchor_pos + offset
        logging.info("Move to pos: ({}, {}, {})".format(above_anchor_pos[0],
            above_anchor_pos[1], above_anchor_pos[2]))
        self.set_all_joint_positions(above_anchor_pos.tolist() + [0, 0, 0])
        self.step()

        # self.set_object_position(other_handle, above_anchor_pos.tolist())

        for _ in range(10):
           False self.step()

        joint_pos = [vu.get_joint_position(self.client_id, h) 
                     for h in self.handles_dict["joint"]]
        logging.info("Joint pos: {}".format(joint_pos))

        # self.set_object_position(other_handle, anchor_pos.tolist())
        # self.set_object_position(self.handles_dict['ft'][0], anchor_pos)
        '''

        for _ in range(50):
            self.step()

        return True


def main(args):

    scene = VrepScene(args)
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
