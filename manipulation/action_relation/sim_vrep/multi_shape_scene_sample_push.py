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
import torch

from utilities import vrep_utils as vu
from lib import vrep

from multi_shape_scene_2 import VrepScene
from utilities.math_utils import get_xy_unit_vector_for_angle
from utilities.math_utils import Sat3D, sample_from_edges, are_position_similar
from utilities.math_utils import sample_exactly_around_edge
from utilities.math_utils import sample_exactly_outside_edge

from utils.colors import bcolors
from action_relation.trainer.train_voxels_online_contrastive import create_voxel_trainer_with_checkpoint

npu = np.random.uniform
npri = np.random.randint
pi = math.pi

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class VrepSampleScene(VrepScene):
    def __init__(self, args):
        super(VrepSampleScene, self).__init__(args)

        self.voxel_trainer = None
    
    def set_model_for_checkpoint(self, checkpoint_path):
        self.voxel_trainer = create_voxel_trainer_with_checkpoint(
            checkpoint_path, cuda=False)
        
    def get_model_prediction_for_data(self, img_data_path, voxel_path):
        trainer = self.voxel_trainer
        dataloader = trainer.dataloader

        data = dataloader.get_img_info_and_voxel_object_for_path(
            img_data_path, voxel_path)
        if data is None:
            print(bcolors.c_yellow("Could not parse voxel data"))
            return None

        img_info, voxel_obj = data[0], data[1]
        status, voxels = voxel_obj.parse()
        # if parse returns false voxels should be None
        assert status

        # ==== Visualize ====
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x,y,z = voxels[0, ...].nonzero()
        # ax.scatter(x, y, z)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.show()

        voxels_tensor = torch.Tensor(voxels)
        proc_voxels_tensor = trainer.process_raw_voxels(voxels_tensor)
        action = torch.Tensor([img_info['action']])

        delta_pose_tensor, delta_pose_numpy = trainer.predict_model_on_data(
            proc_voxels_tensor, action)
        return delta_pose_tensor, delta_pose_numpy

    
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

    def sample_edge_location_for_other_obj(self, anchor_args, anchor_pos,
        anchor_lb, anchor_ub, other_obj_size, outside_xyz, 
        sample_axes_reg):
        '''Sample point on one of the edges of the cuboid.
        
        sample_axes_reg: List of len 3 with values 'in', 'out', 'edge' 
            where 0: in, 1: edge, 2:out.
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
                # Sample exactly outside the edge
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

    def get_other_obj_position(self, anchor_args, anchor_pos, anchor_lb,
                               anchor_ub):
        '''Get position of the other object around the main object.

        anchor_lb: Absolute lower bound of bounding box.
        anchor_ub: Absolute upper bound of bounding box.
        '''
        sample_region_x = (anchor_lb[0]-0.1, anchor_ub[0]+0.1)
        sample_region_y = (anchor_lb[1]-0.1, anchor_ub[1]+0.1)
        max_tries, idx = 100, 0
        scene_on_top = npu(0, 1) < 0.25
        other_obj_size = [npu(0.025, 0.08) for i in range(2)] + [npu(0.025, 0.08)]
        if scene_on_top:
            x = npu(anchor_lb[0], anchor_ub[0])
            y = npu(anchor_lb[1], anchor_ub[1])
            other_obj_ht = 0.25
        else:

            # There are specific cases for this 2D data (since z = 0).
            # First is our object is "out on x-axis" and "in on y-axis"
            # Second is our object is "in on x-axis" and "out on y-axis"
            # Last is our object is "out on x-axis" and "out on y-axis"

            # NOTE: However, in has two parts, completely in and edge, thus we 
            # end up having more cases.
            # sample_axes_reg : 0: in, 1: edge, 2:out
            sample_axes_reg = [np.random.choice([0, 1, 2]), 
                               np.random.choice([0, 1, 2]),
                               0]

            # Make sure that atleast one of the dimensions wouldn't collide.
            while not np.any(np.int32([sample_axes_reg]) != 2):
                sample_axes_reg = [np.random.choice([0, 1, 2]), 
                                   np.random.choice([0, 1, 2]),
                                   0]
            # Get the outside xyz axes
            outside_xyz = []
            for ax in sample_axes_reg:
                outside_val = True if ax == 2 else False
                outside_xyz.append(outside_val)

            other_obj_ht = 0.0
            [x, y, z], sample_edge_info = self.sample_edge_location_for_other_obj(
                anchor_args, anchor_pos, anchor_lb, anchor_ub,
                other_obj_size, outside_xyz, 
                sample_axes_reg,
                )

        other_obj_args = self.get_other_object_args(
            [x, y, other_obj_ht], add_obj_z=True, obj_size=other_obj_size)
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
    
    def run_scene_to_check_falls(self, scene_i, save_dir):
        self.scene_info = {}
        args = self.args

        logging.info("Start sim: {}".format(scene_i))
        self.reset()
        self.step()

        # Generate the anchor object
        anchor_obj_args = self.get_anchor_object_args()
        anchor_handle = self.generate_object_with_args(anchor_obj_args)
        self.handles_dict['anchor_obj'] = anchor_handle
        anchor_pos_dict = dict(
            p=self.get_object_position(anchor_handle),
            q=self.get_object_orientation(anchor_handle)
        )

        # This is a static object which will not move much, hence we do not
        # need to simulate the physics for long.
        for _ in range(10):
            self.step()

        # Generate the other object
        anchor_abs_bb = self.get_absolute_bb(self.handles_dict['anchor_obj'])
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        status, other_obj_info = self.get_other_obj_position(
            anchor_obj_args, anchor_pos_dict['p'], anchor_lb, anchor_ub,
        )
        if not status:
            return False
        other_obj_args = other_obj_info['other_args']
        other_handle = other_obj_info['handle']
        other_pos_dict = dict(
            p=self.get_object_position(other_handle),
            q=self.get_object_orientation(other_handle)
        )
        self.handles_dict['other_obj'] = anchor_handle
        other_obj_x,  other_obj_y, other_obj_z = \
            other_pos_dict['p'][0], other_pos_dict['p'][1], other_pos_dict['p'][2]
        other_lb, other_ub = self.get_object_bb(other_handle)

        # The other object is dynamic and is placed somewhere near to the first
        # object. We will need to simulate it's physics to let it settle down.
        for _ in range(50):
            self.step()

        # Now push the other object towards the anchor object. This should
        # simulate interesting interactions between the two objects.
        self.push_other_obj_towards_anchor(
            anchor_handle, other_handle,
            np.array(anchor_pos_dict['p']),
            np.array(other_pos_dict['p']),
            other_lb,
            other_ub)
        logging.debug("Before push obj dist: {:.4f}".format(
            self.get_object_distance()[-1]))

        for step in range(100):
            self.step()
            if step == 0:
                self.update_keep_object_in_place()

        logging.debug("After push obj dist: {:.4f}".format(
            self.get_object_distance()[-1]))

        # Should we resave and update the position of the other object? This
        # would help in the case when we are spawning new objects.
        other_pos_dict = dict(
            p=self.get_object_position(other_handle),
            q=self.get_object_orientation(other_handle)
        )
        other_obj_x,  other_obj_y, other_obj_z = \
            other_pos_dict['p'][0], other_pos_dict['p'][1], other_pos_dict['p'][2]

        did_add_new_obj = True

        self.scene_info['anchor_args'] = anchor_obj_args
        self.scene_info['other_args'] = other_obj_args

        a_idx = 0
        self.scene_info[a_idx] = {}
        # Wait for sometime
        for _ in range(10):
            self.step()

        _, first_obj_data_dict = self.get_all_objects_info()
        self.update_keep_object_in_place()
        logging.info("Will try to keep objects in place.")
        for _ in range(10):
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
        before_contact_info = self.get_contacts_info()
        voxels_before_dict = self.run_save_voxels_before(
            anchor_handle, other_handle
        )
        self.step()

        logging.info("==== Remove Anchor object ====")
        self.remove_anchor_object()
        # wait and observe effect
        for _ in range(100):
            self.step()

        # Get final pose.
        _, third_obj_data_dict = self.get_other_objects_info()
        logging.info("After removing anchor obj pos change: {:.4f}, {:.4f}, {:.4f}".format(
            third_obj_data_dict['other_pos'][0]-second_obj_data_dict['other_pos'][0],
            third_obj_data_dict['other_pos'][1]-second_obj_data_dict['other_pos'][1],
            third_obj_data_dict['other_pos'][2]-second_obj_data_dict['other_pos'][2]))

        obj_in_place = self.are_pos_and_orientation_similar(
            second_obj_data_dict, third_obj_data_dict, 
            verify_other_obj_only=True)
        self.scene_info[a_idx]['obj_in_place_after_anchor_remove'] = obj_in_place
        self.scene_info[a_idx]['after'] = third_obj_data_dict
        if not obj_in_place:
            logging.info("Objects changed position AFTER removing anchor.")

        recorded_contact_info = []
        self.step()

        after_contact_info = None
        voxels_after_dict = None

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
        logging.info("Contact len: before: {}".format(before_contact_len))
        logging.info(" ==== Scene {} Done ====".format(scene_i))

        if args.reset_every_action == 1:
            self.reset()

        return True

    def run_scene(self, scene_i, save_dir):
        self.scene_info = {}
        args = self.args

        logging.info("Start sim: {}".format(scene_i))
        self.reset()
        self.step()

        # Generate the anchor object
        anchor_obj_args = self.get_anchor_object_args()
        anchor_handle = self.generate_object_with_args(anchor_obj_args)
        self.handles_dict['anchor_obj'] = anchor_handle
        anchor_pos_dict = dict(
            p=self.get_object_position(anchor_handle),
            q=self.get_object_orientation(anchor_handle)
        )

        # This is a static object which will not move much, hence we do not
        # need to simulate the physics for long.
        for _ in range(10):
            self.step()

        # Generate the other object
        anchor_abs_bb = self.get_absolute_bb(self.handles_dict['anchor_obj'])
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        status, other_obj_info = self.get_other_obj_position(
            anchor_obj_args, anchor_pos_dict['p'], anchor_lb, anchor_ub,
        )
        if not status:
            logging.info("Other object not added. Will return.")
            return False

        other_obj_args = other_obj_info['other_args']
        other_handle = other_obj_info['handle']
        other_pos_dict = dict(
            p=self.get_object_position(other_handle),
            q=self.get_object_orientation(other_handle)
        )
        self.handles_dict['other_obj'] = anchor_handle
        other_obj_x,  other_obj_y, other_obj_z = \
            other_pos_dict['p'][0], other_pos_dict['p'][1], other_pos_dict['p'][2]
        other_lb, other_ub = self.get_object_bb(other_handle)

        # The other object is dynamic and is placed somewhere near to the first
        # object. We will need to simulate it's physics to let it settle down.
        for _ in range(50):
            self.step()

        # Now push the other object towards the anchor object. This should
        # simulate interesting interactions between the two objects.
        self.push_other_obj_towards_anchor(
            anchor_handle, other_handle,
            np.array(anchor_pos_dict['p']),
            np.array(other_pos_dict['p']),
            other_lb,
            other_ub)
        logging.debug("Before push obj dist: {:.4f}".format(
            self.get_object_distance()[-1]))

        for step in range(100):
            self.step()
            if step == 0:
                self.update_keep_object_in_place()

        logging.debug("After push obj dist: {:.4f}".format(
            self.get_object_distance()[-1]))

        # Should we resave and update the position of the other object? This
        # would help in the case when we are spawning new objects.
        other_pos_dict = dict(
            p=self.get_object_position(other_handle),
            q=self.get_object_orientation(other_handle)
        )
        other_obj_x,  other_obj_y, other_obj_z = \
            other_pos_dict['p'][0], other_pos_dict['p'][1], other_pos_dict['p'][2]
        # other_lb, other_ub = self.get_object_bb(other_handle)

        action_angles = [get_xy_unit_vector_for_angle((t*pi)/4.0)
                         for t in range(0, 8, 2)]
        action_mag = 8
        action_vec = [(action_mag*np.array(ang)).tolist()
                      for ang in action_angles]
        # import ipdb; ipdb.set_trace()

        # actions = [
        #     (-5,0), (-8,0), (8,0), (5,0),
        #     (0,-8), (0,-5), (0,8), (0,5)
        # ]
        did_add_new_obj = True

        self.scene_info['anchor_args'] = anchor_obj_args
        self.scene_info['other_args'] = other_obj_args

        for a_idx, a in enumerate(action_vec):
            logging.info(" ==== Scene {} action: {} [{}/{}] start ====".format(
                scene_i, a, a_idx, len(action_vec)))
            self.scene_info[a_idx] = {
                'action': a,
            }
            bad_sample = False
            self.step()
            if args.reset_every_action == 1 and a_idx > 0:
                # Create a scene from scratch
                anchor_handle = self.generate_object_with_args(
                    anchor_obj_args)
                self.handles_dict['anchor_obj'] = anchor_handle
                for _ in range(10):
                    self.step()
                other_handle = self.generate_object_with_args(
                    other_obj_args)
                self.handles_dict['other_obj'] = other_handle
                did_add_new_obj = False

                self.push_other_obj_towards_anchor(
                    anchor_handle, other_handle,
                    np.array(anchor_pos_dict['p']),
                    np.array(other_pos_dict['p']),
                    other_lb,
                    other_ub)

                for step in range(100):
                    self.step()
                    if step == 0:
                        self.update_keep_object_in_place()

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

            # Wait for sometime
            for _ in range(10):
                self.step()

            _, first_obj_data_dict = self.get_all_objects_info()
            self.update_keep_object_in_place()
            logging.info("Will try to keep objects in place.")
            for _ in range(10):
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
            before_contact_info = self.get_contacts_info()
            voxels_before_dict = self.run_save_voxels_before(
                anchor_handle, other_handle
            )
            self.toggle_recording_contact_data()
            self.step()

            logging.info("==== Apply Force ====")
            # push
            # action_status = self.apply_force([a[0], a[1], a[2]])
            if args.action_type == 'world_frame':
                action_status = self.apply_force_in_ref_frame([a[0], a[1], a[2]])
            elif args.action_type == 'obj_frame':
                action_status = self.apply_force_on_object(
                    other_handle, [a[0], a[1], a[2]])
            else:
                raise ValueError("Invalid action type: {}".format(args.action_type))
            if not action_status:
                bad_sample = True

            # wait and observe effect
            for _ in range(100):
                self.step()

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
            _, recorded_contact_info = self.save_recorded_contact_data()
            self.step()

            after_contact_info = self.get_contacts_info()
            voxels_after_dict = self.run_save_voxels_after(
                anchor_handle, other_handle)

            if not bad_sample:
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
            
            if len(args.voxel_model_checkpoint) > 0:
                # Verify the output of the model
                img_data_path = os.path.join(save_dir, '{}_img_data.pkl'.format(a_idx))
                voxel_path = os.path.join(save_dir, '{}_voxel_data.pkl'.format(a_idx))
                result = self.get_model_prediction_for_data(img_data_path, voxel_path)
                if result is None:
                    print("Cannot parse input")
                else:
                    _, pred_pos_arr  = result[0], result[1]
                    delta_pos = [
                        third_obj_data_dict['other_pos'][i] -
                        second_obj_data_dict['other_pos'][i] for i in range(3)]
                    gt_pos_arr = np.array(delta_pos)
                    pred_pos_arr = pred_pos_arr[0, :3]
                    l2_dist = np.linalg.norm(gt_pos_arr-pred_pos_arr, ord=2)
                    l1_dist = np.linalg.norm(gt_pos_arr-pred_pos_arr, ord=1)
                    print(bcolors.c_red("   \t  GT:   \t    {}\n"
                                        "   \t  Pred: \t    {}\n" 
                                        "   \t  Diff: \t    {}\n"
                                        "   \t  L1:   \t    {}\n"
                                        "   \t  L2:   \t    {}\n".format(
                                            gt_pos_arr, pred_pos_arr,
                                            gt_pos_arr - pred_pos_arr,
                                            l1_dist, l2_dist
                                        )))

            if args.reset_every_action == 1:
                self.reset()

        return True


def main(args):
    scene = VrepSampleScene(args)
    num_scenes, total_try = 0, 0
    if len(args.voxel_model_checkpoint) > 0:
        scene.set_model_for_checkpoint(args.voxel_model_checkpoint)

    while num_scenes < args.num_scenes:
        save_path = os.path.join(args.save_dir, str(num_scenes))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.remove_anchor:
            status = scene.run_scene_to_check_falls(num_scenes, save_path)
        else:
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
    parser.add_argument('--action_type', type=str, default='obj_frame',
                        choices=['world_frame', 'obj_frame'],
                        help='Action to be performed in world frame or obj frame')
    parser.add_argument('--remove_anchor', type=str2bool, default=False, 
                        help='Remove anchor from scene.')
    parser.add_argument('--voxel_model_checkpoint', type=str, default='', 
                        help='Model checkpoint to predict action effects.')
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    main(args)
