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
from lib import vrep

from multi_shape_scene_2 import VrepScene
from utilities.math_utils import get_xy_unit_vector_for_angle

npu = np.random.uniform
npri = np.random.randint
pi = math.pi

INIT_POS_LIST = [0, 0, 4]
INIT_POS_ARR = np.array(INIT_POS_LIST)


class VrepSimpleNearScene(VrepScene):
    def __init__(self, args):
        super(VrepSimpleNearScene, self).__init__(args)

        if args.sample_on_top == 1:
            self.init_pos_list = [0, 0, 4]
        else:
            self.init_pos_list = [0, 0, 2]
        self.init_pos_arr = np.array(self.init_pos_list)

    def get_force_vector_from_other_to_anchor(self):
        # ms_GetForceVectorFromOtherToAnchor
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "WorldCreator",
            vrep.sim_scripttype_childscript,
            "ms_GetForceVectorFromOtherToAnchor",
            [],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            logging.error("Error in getting force vec: {}".format(res[0]))
        print(np.array(res[2]))
        return np.array(res[2])

    def get_new_other_obj_position(self, anchor_args, anchor_pos, anchor_lb,
                                   anchor_ub, other_handle, oriented_other_lb,
                                   oriented_other_ub):
        '''Get position of the other object around the main object.

        anchor_lb: Relative lower bound of bounding box.
        anchor_ub: Relative upper bound of bounding box.
        oriented_other_lb: Absolute lower bound of bounding box in world frame.
        oriented_other_ub: Absolute upper bound of bounding box in world frame.
        '''
        args = self.args
        rel_pos = npri(1, 4)
        oriented_other_center = (oriented_other_lb + oriented_other_ub) / 2
        oriented_bb_size = (oriented_other_ub - oriented_other_lb)
        oriented_other_center = oriented_other_center - INIT_POS_ARR
        other_obj_ht = oriented_other_ub[-1] - oriented_other_lb[-1]

        if args.sample_on_top == 1:
            th = -0.05
            ht = anchor_pos[-1] + anchor_ub[-1] + other_obj_ht/2.0
        else:
            th = 0.01
            ht = other_obj_ht/2.0
        logging.info("Other obj new ht: {:.3f}".format(ht))

        if rel_pos == 0:    # Left
            new_y = npu(anchor_lb[1], anchor_ub[1])
            new_pos = [anchor_lb[0] - oriented_bb_size[0]/2.0-th, new_y, ht]
        elif rel_pos == 1:  # right
            new_y = npu(anchor_lb[1], anchor_ub[1])
            new_pos = [anchor_ub[0] + oriented_bb_size[0]/2.0+th, new_y, ht]
        elif rel_pos == 2:  # infront
            new_x = npu(anchor_lb[0], anchor_ub[0])
            new_pos = [new_x, anchor_lb[1] - oriented_bb_size[1]/2.0-th, ht]
        elif rel_pos == 3:  # back
            new_x = npu(anchor_lb[0], anchor_ub[0])
            new_pos = [new_x, anchor_ub[1] + oriented_bb_size[1]/2.0+th, ht]
        else:
            raise ValueError("Invalid rel_pos: {}".format(rel_pos))

        logging.info("Anchor lb: {},  \t  ub: {}".format(anchor_lb, anchor_ub))
        # import ipdb; ipdb.set_trace()

        # Set new object position
        resp = vrep.simxSetObjectPosition(
            self.client_id, other_handle, -1, new_pos,
            vrep.simx_opmode_blocking)
        if resp != vrep.simx_return_ok:
            logging.error("Cannot set new obj position: {}".format(resp))
        return resp == vrep.simx_return_ok, new_pos

    def get_initial_other_obj_position(self, anchor_args, anchor_pos,
                                       anchor_lb, anchor_ub):
        obj_size = [npu(0.08, 0.4) for i in range(3)]
        other_obj_args = self.get_other_object_args(
            self.init_pos_list, add_obj_z=True, obj_size=obj_size)
        other_handle = self.generate_object_with_args(other_obj_args)
        info_dict = dict(
            other_args=other_obj_args,
            handle=other_handle,
        )
        return True, info_dict

    def push_other_obj_towards_anchor(self, anchor_handle, other_handle,
                                      anchor_pos, other_pos, other_lb,
                                      other_ub):
        force_pos = (other_lb + other_ub) / 2.0
        force_pos[-1] = other_ub[-1]
        # Try to force other object into the ground
        anchor_pos[-1] = -1
        force_dir_vec = (other_pos - anchor_pos) + force_pos
        # We don't really want a force in the z-direction.
        force_dir_vec = -force_dir_vec/np.linalg.norm(force_dir_vec)
        force_mag = 10

        force_dir_vec_2 = self.get_force_vector_from_other_to_anchor()
        force_dir_vec_2 = force_dir_vec_2 / np.linalg.norm(force_dir_vec_2)

        # Add force at pos=ub, magnitude = 10 * force_dir_vec
        # import ipdb; ipdb.set_trace()
        status = self.apply_force(
            other_handle,
            force_pos,
            force_mag * force_dir_vec_2,
            )

    def run_scene(self, scene_i, save_dir):
        self.scene_info = {}
        args = self.args

        logging.info("Start sim: {}".format(scene_i))
        self.reset()
        self.step()

        # Generate the anchor object
        anchor_args = self.get_anchor_object_args()
        anchor_handle = self.generate_object_with_args(anchor_args)
        self.handles_dict['anchor_obj'] = anchor_handle
        anchor_pos_dict = dict(
            p=self.get_object_position(anchor_handle),
            q=self.get_object_orientation(anchor_handle)
        )

        for _ in range(100):
            self.step()

        # Generate the other object
        anchor_abs_bb = self.get_absolute_bb(self.handles_dict['anchor_obj'])
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        anchor_rel_lb, anchor_rel_ub = self.get_object_bb(
            self.handles_dict['anchor_obj'])
        status, other_obj_info = self.get_initial_other_obj_position(
            anchor_args, anchor_pos_dict['p'], anchor_lb, anchor_ub)
        other_obj_args = other_obj_info['other_args']
        other_handle = other_obj_info['handle']
        if not status:
            return False

        # Now find space to put the object right next to the anchor object
        oriented_other_lb, oriented_other_ub = self.get_oriented_bounding_box()
        status, new_pos = self.get_new_other_obj_position(
            anchor_args, anchor_pos_dict['p'], anchor_rel_lb, anchor_rel_ub,
            other_handle, oriented_other_lb, oriented_other_ub)
        logging.info("Position near the anchor: {}".format(new_pos))
        if not status:
            logging.info("Cannot place object in new location. Sample again!!")
            return
        other_obj_args['pos'] = new_pos
        status = self.remove_other_object()
        if not status:
            logging.error("Cannot remove other object")
            return False
        other_handle = self.generate_object_with_args(other_obj_args)
        other_obj_info['handle'] = other_handle

        self.update_keep_object_in_place()
        for s in range(100):
            self.step()
            if s == 0:
                self.update_keep_object_in_place()

        self.handles_dict['other_obj'] = other_handle
        other_pos_dict = dict(
            p=self.get_object_position(other_handle),
            q=self.get_object_orientation(other_handle)
        )
        other_obj_x,  other_obj_y, other_obj_z = \
            other_pos_dict['p'][0], other_pos_dict['p'][1], other_pos_dict['p'][2]
        other_lb, other_ub = self.get_object_bb(other_handle)

        action_angles = [get_xy_unit_vector_for_angle((t*pi)/2.0)
                         for t in range(0, 4)]
        action_mag = 8
        action_vec = [(action_mag*np.array(ang)).tolist()
                      for ang in action_angles]
        # actions = [
        #     (-5,0), (-8,0), (8,0), (5,0),
        #     (0,-8), (0,-5), (0,8), (0,5)
        # ]
        did_add_new_obj = True

        self.scene_info['anchor_args'] = anchor_args
        self.scene_info['other_args'] = other_obj_args

        for a_idx, a in enumerate(action_vec):
            logging.info(" ==== Scene {} action: {} start ====".format(
                scene_i, a))
            self.scene_info[a_idx] = {
                'action': a,
            }
            bad_sample = False
            self.toggle_keep_object_in_place()
            self.step()
            if args.reset_every_action == 1 and a_idx > 0:
                # Create a scene from scratch
                anchor_handle = self.generate_object_with_args(
                    anchor_args)
                self.handles_dict['anchor_obj'] = anchor_handle
                for _ in range(100):
                    self.step()
                other_handle = self.generate_object_with_args(
                    other_obj_args)
                self.handles_dict['other_obj'] = other_handle
                did_add_new_obj = False

                self.update_keep_object_in_place()
                for step in range(100):
                    self.step()
                    if step == 0:
                        self.update_keep_object_in_place()

            elif a_idx > 0:
                # Set objects to their respective positions directly.
                self.set_object_position(anchor_handle, anchor_pos_dict['p'])
                self.set_object_orientation(anchor_handle, anchor_pos_dict['q'])
                for _ in range(50):
                    self.step()
                logging.debug("Other obj : {:.4f}, {:.4f}, {:.4f}".format(
                    other_obj_x, other_obj_y, other_obj_z
                ))
                self.set_object_position(
                    other_handle, [other_obj_x, other_obj_y, other_obj_z])
                self.set_object_orientation(other_handle, other_obj_angles)
                did_add_new_obj = False

            # Wait for sometime
            for _ in range(100):
                self.step()

            self.update_keep_object_in_place()
            logging.info("Will try to keep objects in place.")
            for _ in range(10):
                self.step()

            _, first_obj_data_dict = self.get_all_objects_info()

            # Now simulate the physics engine for sometime
            for _ in range(100):
                self.step()


            _, second_obj_data_dict = self.get_all_objects_info()

            obj_in_place = self.are_pos_and_orientation_similar(
                first_obj_data_dict, second_obj_data_dict)

            self.debug_initial_final_position(
                first_obj_data_dict, second_obj_data_dict)

            if not obj_in_place:
                logging.error("OBJECTS STILL MOVING!! Will sample again.")
                return False
            if not self.are_objects_close(second_obj_data_dict):
                logging.error("Objects are not close. Will sample again.")
                return False

            self.scene_info[a_idx]['before'] = second_obj_data_dict
            self.scene_info[a_idx]['before_dist'] = self.get_object_distance()[-1]

            for _ in range(5):
                self.step()

            # Get image and octree data
            voxels_before_dict = self.run_save_voxels_before(anchor_handle, 
                                                             other_handle)
            if self.args.toggle_keep_obj_in_place_before_force == 1:
                self.toggle_keep_object_in_place()
            self.step()

            #  Get contact info
            before_contact_info = self.get_contacts_info()
            self.toggle_recording_contact_data()

            for _ in range(5):
                self.step()

            logging.info("==== Apply Force ====")
            # push
            # action_status = self.apply_force([0, 0, 0], [a[0], a[1], 0])
            action_status = self.apply_force_on_object(
                other_handle, [a[0], a[1], a[2]])
            if not action_status:
                bad_sample = True

            # wait and observe effect
            for _ in range(100):
                self.step()

            if self.args.toggle_keep_obj_in_place_before_force == 1:
                self.toggle_keep_object_in_place()

            logging.info("After force obj dist: {:.4f}".format(
                self.get_object_distance()[-1]))

            # Get final pose.
            _, third_obj_data_dict = self.get_all_objects_info()
            obj_in_place = self.are_pos_and_orientation_similar(
                second_obj_data_dict, third_obj_data_dict)
            self.debug_initial_final_position(
                second_obj_data_dict, third_obj_data_dict)
            self.scene_info[a_idx]['after'] = third_obj_data_dict
            if not obj_in_place:
                logging.info("Objects changed position AFTER push.")

            self.scene_info[a_idx]['after_dist'] = self.get_object_distance()[-1]

            self.toggle_recording_contact_data()
            recorded_contact_info = self.save_recorded_contact_data()
            self.step()

            voxels_after_dict = self.run_save_voxels_after(
                anchor_handle, other_handle)
            for _ in range(5):
                self.step()

            if not bad_sample:
                self.save_scene_data(
                    save_dir, a_idx, self.scene_info,
                    voxels_before_dict,
                    voxels_after_dict,
                    before_contact_info,
                    after_contact_info,
                    recorded_contact_info,
                )

            logging.info(" ==== Scene {} action: {} Done ====".format(
                scene_i, a))

            if args.reset_every_action == 1:
                self.reset()

        return True


def main(args):
    scene = VrepSimpleNearScene(args)
    num_scenes, total_try = 0, 0
    while num_scenes < args.num_scenes:
        save_path = os.path.join(args.save_dir, str(num_scenes))
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
    parser.add_argument('--sample_on_top', type=int, default=0, choices=[0, 1],
                        help='Should we sample on or overlapping the anchor '
                             'object.')
    parser.add_argument('--toggle_keep_obj_in_place_before_force', type=int,
                        default=0, choices=[0, 1],
                        help='Toggle keep object in place before applying force'
                             'manually.')
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    main(args)
