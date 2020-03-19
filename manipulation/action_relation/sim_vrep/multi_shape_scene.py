import numpy as np
import argparse
import os
import pickle
import json
import pprint
import cv2
import shutil
import logging

from utilities import vrep_utils as vu

from lib import vrep

npu = np.random.uniform
npri = np.random.randint

DT = 0.001
DEFAULT_DT = 0.05


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
            vrep.simx_opmode_oneshot_wait)
        assert ret == vrep.simx_return_ok, "Cannot set sim timestep"

    def stop(self):
        vu.stop_sim(self.client_id)

    def stop_and_finish(self):
        self.stop()
        vrep.simxFinish(self.client_id)

    def reset(self):
        self.stop()
        self.start()

    def connect(self, scene_file, port):
        client_id = vu.connect_to_vrep(port=port)
        res = vu.load_scene(client_id, scene_file)
        return client_id

    def start(self):
        vu.start_sim(self.client_id)

    def step(self):
        vu.step_sim(self.client_id)

    def get_handles(self):
        cam_rgb_names = ["Camera"]
        cam_rgb_handles = [vrep.simxGetObjectHandle(
            self.client_id, n, vrep.simx_opmode_blocking)[1] for n in cam_rgb_names]

        octree_handle = vu.get_handle_by_name(self.client_id, 'Octree')
        vision_sensor = vu.get_handle_by_name(self.client_id, 'Vision_sensor')

        return dict(
            octree=octree_handle,
            cam_rgb=cam_rgb_handles,
            vision_sensor=vision_sensor,
        )

    def get_absolute_bb(self, obj_handle):
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

    def apply_force_on_object(self, o_handle, force):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_ApplyForceOnObject",
            [int(self.handles_dict['other_obj'])],
            [force[0], force[1], force[2]],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in apply force on object: {}".format(res[0]))
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
        assert len(res[2]) == 24, "Got invalid number of return values"
        d = {}
        d['anchor_pos'] = res[2][:3]
        d['anchor_q'] = res[2][3:6]
        d['other_pos'] = res[2][6:9]
        d['other_q'] = res[2][9:12]
        d['anchor_bb'] = res[2][12:18]
        d['other_bb'] = res[2][18:24]

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
            # q[0], q[1], q[2],
            0, 0, 0,
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
        obj_type = npri(1, 4)
        obj_type = 2
        obj_size = [npu(0.02, 0.4) for i in range(3)]
        pos = [0, 0, obj_size[-1]/2.0]  # Create above ground so that it falls down
        q = [npu(-90.0, 90.0) for i in range(3)]
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
            is_static=0,
            lin_damp=0.1,
            ang_damp=0.1,
        )

    def get_other_object_args(self, pos, add_obj_z=True):
        obj_type = npri(1, 4)
        obj_type = 2
        obj_size = [npu(0.02, 0.4) for i in range(3)]
        if add_obj_z:
            pos[-1] = pos[-1] + obj_size[-1]/2.0
        q = [npu(-90.0, 90.0) for i in range(3)]
        color = [0.0, 1.0, 1.0]
        mass = 1.0  # Heavy object

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

    def run_scene(self, i, save_dir):
        self.scene_info = {}
        args = self.args

        logging.info("Start sim: {}".format(i))
        self.reset()
        self.step()
        other_obj_ht = 0.1
        other_obj_size = [0.1, 0.1, 0.1]

        # TODO: Generate the anchor object
        anchor_obj_args = self.get_anchor_object_args()
        anchor_handle = self.generate_object_with_args(anchor_obj_args)
        self.handles_dict['anchor_obj'] = anchor_handle
        anchor_pos = self.get_object_position(anchor_handle)
        anchor_q = self.get_object_orientation(anchor_handle)
        for _ in range(100):
            self.step()

        # TODO: Generate the other object

        # Find the position for the other object
        anchor_abs_bb = self.get_absolute_bb(self.handles_dict['anchor_obj'])
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        logging.debug("Abs bb: \n"
                      "\t    lb: \t  {}\n"
                      "\t    ub: \t  {}\n".format(anchor_lb, anchor_ub))

        low_x = anchor_lb[0] - 0.02
        low_y = anchor_lb[1] - 0.02
        high_x = anchor_ub[0] + 0.02
        high_y = anchor_ub[1] + 0.02
        logging.debug("low: ({},{}), high: ({},{})".format(
            low_x, low_y, high_x, high_y))

        other_obj_x = npu(low_x, high_x)
        other_obj_y = npu(low_y, high_y)
        other_obj_size = [npu(0.02, 0.4) for i in range(3)]
        other_obj_z = anchor_ub[2]
        other_obj_args = self.get_other_object_args(
            [other_obj_x, other_obj_y, other_obj_z], add_obj_z=True)
        other_obj_handle = self.generate_object_with_args(other_obj_args)
        self.handles_dict['other_obj'] = other_obj_handle

        other_obj_angles = self.get_object_orientation(
            self.handles_dict['other_obj'])

        actions = [
            (-15,0), (-20,0), (20,0), (15,0),
            (0,-15), (0,-20), (0,20), (0,15)
        ]

        anchor_handle = self.handles_dict['anchor_obj']
        other_handle = self.handles_dict['other_obj']
        did_add_new_obj = True

        for a_idx, a in enumerate(actions):
            logging.info(" ==== Scene {} action: {} start ====".format(i, a))
            self.scene_info[a_idx] = {
                'action': a,
            }
            bad_sample = False

            if args.reset_every_action == 1 and a_idx > 0:
                # Create a scene from scratch
                anchor_handle = self.generate_object_with_args(
                    anchor_obj_args)
                self.handles_dict['anchor_obj'] = anchor_handle
                self.update_add_object(0)
                for _ in range(100):
                    self.step()
                other_handle = self.generate_object_with_args(
                    other_obj_args)
                self.handles_dict['other_obj'] = other_handle
                self.update_add_object(1)
                did_add_new_obj = False
            elif a_idx > 0:
                # Set objects to their respective positions directly.
                self.set_object_position(anchor_handle, anchor_pos)
                self.set_object_orientation(anchor_handle, anchor_q)
                for _ in range(50):
                    self.step()
                logging.debug("Other obj : {:.4f}, {:.4f}, {:.4f}".format(
                    other_obj_x, other_obj_y, other_obj_z
                ))
                self.set_object_position(
                    other_handle, [other_obj_x, other_obj_y, other_obj_z])
                self.set_object_orientation(other_handle, other_obj_angles)
                did_add_new_obj = False

            if did_add_new_obj:
                lin_damp, ang_damp = self.get_object_damping_param(anchor_handle)
                logging.info("Lin: {:.4f}, ang: {:.4f}".format(lin_damp, ang_damp))

            # Wait for sometime
            for _ in range(50):
                self.step()

            self.update_keep_object_in_place()
            logging.info("Will try to keep objects in place.")

            # Try to stop the motion for both, anchor and other object
            # logging.debug("Will try to stop anchor: ")
            # for _ in range(10):
            #     self.stop_object_controller(anchor_handle, DT)
            #     self.step()
            lin_v, ang_v = self.get_object_velocity(
                anchor_handle, streaming=False)
            lin_v_arr, ang_v_arr = np.array(lin_v), np.array(ang_v)
            logging.info("Anchor obj vel lin: {}, ang: {}".format(
                lin_v_arr, ang_v_arr))

            if np.any(lin_v_arr > 0.001) or np.any(ang_v_arr > 0.001):
                logging.error("Anchor obj STILL MOVING")
                # return False

            # logging.debug("Will try to stop other: ")
            # for _ in range(10):
            #     self.stop_object_controller(other_handle, DT)
            #     self.step()
            lin_v, ang_v = self.get_object_velocity(
                other_handle, streaming=False)
            lin_v_arr, ang_v_arr = np.array(lin_v), np.array(ang_v)
            logging.info("Other obj vel lin: {}, ang_v: {}".format(
                lin_v_arr, ang_v_arr))
            if np.any(lin_v_arr > 0.001) or np.any(ang_v_arr > 0.001):
                logging.error("Other obj STILL MOVING")
                # return False

            # Get initial data
            _, initial_obj_data_dict = self.get_all_objects_info()
            logging.info("Initial pos: \n"
                         "anchor:   pos: {}, \t q: {}    \t   "
                         "other:    pos: {}, \t q: {}".format(
                             np.array(initial_obj_data_dict['anchor_pos']),
                             np.array(initial_obj_data_dict['anchor_q']),
                             np.array(initial_obj_data_dict['other_pos']),
                             np.array(initial_obj_data_dict['other_q'])
                         ))

            self.scene_info[a_idx]['before'] = initial_obj_data_dict

            for _ in range(5):
                self.step()

            # Get image and octree data
            img_data_before = self.get_vision_image()
            self.update_octree()
            for _ in range(10):
                self.step()
            voxels = self.get_octree_voxels()
            logging.debug("Got initial voxels: {}".format(len(voxels)))

            for _ in range(5):
                self.step()
            self.clear_octree()
            for _ in range(5):
                self.step()

            # push
            # action_status = self.apply_force([0, 0, 0], [a[0], a[1], 0])
            action_status = self.apply_force_on_object(
                other_handle, [a[0], a[1], 0])
            if not action_status:
                bad_sample = True

            # wait and observe effect
            for _ in range(100):
                self.step()

            # Get final pose.
            _, final_obj_data_dict = self.get_all_objects_info()
            logging.info("Final: \n"
                         "anchor:   pos: {}, \t q: {}    \t   "
                         "other:    pos: {}, \t q: {}".format(
                             np.array(initial_obj_data_dict['anchor_pos']),
                             np.array(initial_obj_data_dict['anchor_q']),
                             np.array(initial_obj_data_dict['other_pos']),
                             np.array(initial_obj_data_dict['other_q'])
                         ))
            self.scene_info[a_idx]['after'] = final_obj_data_dict

            for _ in range(50):
                self.step()

            img_data_after = self.get_vision_image()

            if not bad_sample:
                cv2.imwrite(os.path.join(
                    save_dir, '{}_before_vision_sensor.png'.format(a_idx)),
                    img_data_before)
                cv2.imwrite(os.path.join(
                    save_dir, '{}_after_vision_sensor.png'.format(a_idx)),
                    img_data_after)
                pkl_data = {'data': self.scene_info[a_idx],
                            'voxels_before': voxels}
                pkl_path = os.path.join(
                    save_dir, '{}_img_data.pkl'.format(a_idx))
                with open(pkl_path, 'wb') as pkl_f:
                    pickle.dump(pkl_data, pkl_f, protocol=2)

                json_path = os.path.join(
                    save_dir, '{}_img_data.json'.format(a_idx))
                with open(json_path, 'w') as json_f: 
                    json_f.write(json.dumps(self.scene_info[a_idx]))
                json_path = os.path.join(
                    save_dir, 'all_img_data.json'.format(a_idx))
                with open(json_path, 'w') as json_f:
                    json_f.write(json.dumps(self.scene_info))

            logging.info(" ==== Scene {} action: {} Done ====".format(i, a))

            if args.reset_every_action == 1:
                self.reset()

        return True


def main(args):

    scene = VrepScene(args)
    num_scenes = 0
    while num_scenes < args.num_scenes:
        save_path = os.path.join(args.save_dir, str(num_scenes))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        status = scene.run_scene(num_scenes, save_path)
        if not status:
            shutil.rmtree(save_path)
            print("Incorrect scene, scenes finished: {}".format(num_scenes))
        else:
            num_scenes += 1
            print("Correct scene: scenes finished: {}".format(num_scenes))
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
    np.set_printoptions(precision=3)
    main(args)
