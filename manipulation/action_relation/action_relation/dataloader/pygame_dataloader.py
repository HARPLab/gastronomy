import numpy as np

import csv
import h5py
import os
import ipdb
import pickle
import glob

from utils import data_utils
from utils import image_utils
from utils.colors import bcolors

import torch
from torchvision.transforms import functional as F


class SimpleImageDataloader(object):
    def __init__(self, config,
                 train_dir=None,
                 test_dir=None):
        self._config = config
        self.transforms = None

        self.train_dir = train_dir \
            if train_dir is not None else config.args.train_dir
        self.test_dir = test_dir \
            if test_dir is not None else config.args.test_dir

        self.train_idx_to_data_dict = self.load_image_data(self.train_dir)
        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))

        self.test_idx_to_data_dict = self.load_image_data(self.test_dir)
        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

    def load_image_data(self, demo_dir):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx = 0
        demo_idx_to_path_dict = {}

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        for root, dirs, files in os.walk(demo_dir, followlinks=True):
            if 'img_data.pkl' in files:
                # We are in a data dir
                with open(os.path.join(root, 'img_data.pkl'), 'rb') as pkl_f:
                    pkl_data = pickle.load(pkl_f)
                steps_before_action = pkl_data['action']['steps_before_action']
                steps_after_action = pkl_data['action']['steps_after_action']
                total_steps = steps_before_action + steps_after_action 
                
                if pkl_data.get('{}_anchor_obj_bb_xyxy'.format(total_steps)) is None:
                    continue

                img_prefix = steps_before_action
                img_suffix = (img_prefix // 5) - 1
                img_path = os.path.join(root, '{}_img_{}.png'.format(
                    img_prefix, img_suffix))
                scene_type = int(os.path.basename(
                    os.path.dirname(root)).split('_')[0])
                pkl_data['scene_type'] = scene_type

                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['img_path'] = img_path
                demo_idx_to_path_dict[demo_idx]['img_info'] = pkl_data

                demo_idx = demo_idx + 1

        return demo_idx_to_path_dict

    def get_data_size(self, train=True):
        if train:
            return len(self.train_idx_to_data_dict)
        else:
            return len(self.test_idx_to_data_dict)

    def get_mask(self, image, color_val):
        mask = np.zeros_like(image, dtype=np.float32)
        # mask[np.where(image == color_val)[1]] = 1
        # mask[image == color_val] = 1
        sum_img = np.sum(np.array(image) == color_val, axis=2)
        mask = sum_img == 3
        return mask

    def get_masked_image_with_bb(self, img, *args):
        mask = np.zeros_like(img, dtype=np.bool)
        for i, bb_xyxy in enumerate(args):
            x1, y1, x2, y2 = bb_xyxy
            mask[y1:y2, x1:x2, :] = 1
        masked_img = img * mask
        return masked_img, mask.astype(np.float32)

    def get_masked_tensor_with_bb(self, img_tensor, *args):
        mask = torch.zeros_like(img_tensor)
        for i, bb_xyxy in enumerate(args):
            x1, y1, x2, y2 = bb_xyxy
            # C, H, W format
            mask[:, y1:y2, x1:x2] = 1
        return img_tensor * mask, mask

    def get_masked_images_for_target_bb(self, img, target_bb,
                                        ground_obstacle_bb_list,
                                        air_obstacle_bb_list):
        # Get all masked images and the masks
        target_air_obstacle_mask_img, target_air_obstacle_mask = [], []
        for i, air_obstacle_bb in enumerate(air_obstacle_bb_list):
            target_obstacle_img, target_obstacle_mask = \
                self.get_masked_image_with_bb(img, target_bb, air_obstacle_bb)
            target_air_obstacle_mask_img.append(target_obstacle_img)
            target_air_obstacle_mask.append(target_obstacle_mask)
        target_ground_obstacle_mask_img, target_ground_obstacle_mask = [], []
        for i, ground_obstacle_bb in enumerate(ground_obstacle_bb_list):
            target_obstacle_img, target_obstacle_mask = \
                self.get_masked_image_with_bb(img, target_bb, ground_obstacle_bb)
            target_ground_obstacle_mask_img.append(target_obstacle_img)
            target_ground_obstacle_mask.append(target_obstacle_mask)
        return {
            'air_obstacle_mask_img': target_air_obstacle_mask_img,
            'air_obstacle_mask': target_air_obstacle_mask,
            'ground_obstacle_mask_img': target_ground_obstacle_mask_img,
            'ground_obstacle_mask': target_ground_obstacle_mask,
        }

    def get_masked_image_tensor_for_target_bb(self, img_tensor, target_bb,
                                              ground_obstacle_bb_list,
                                              air_obstacle_bb_list):
        # Get all masked images and the masks
        target_air_obstacle_mask_img, target_air_obstacle_mask = [], []
        for i, air_obstacle_bb in enumerate(air_obstacle_bb_list):
            target_obstacle_img, target_obstacle_mask = \
                self.get_masked_tensor_with_bb(
                    img_tensor, target_bb, air_obstacle_bb)
            target_air_obstacle_mask_img.append(target_obstacle_img)
            target_air_obstacle_mask.append(target_obstacle_mask)

        target_ground_obstacle_mask_img, target_ground_obstacle_mask = [], []
        for i, ground_obstacle_bb in enumerate(ground_obstacle_bb_list):
            target_obstacle_img, target_obstacle_mask = \
                self.get_masked_tensor_with_bb(
                    img_tensor, target_bb, ground_obstacle_bb)
            target_ground_obstacle_mask_img.append(target_obstacle_img)
            target_ground_obstacle_mask.append(target_obstacle_mask)

        return {
            'air_obstacle_mask_img': target_air_obstacle_mask_img,
            'air_obstacle_mask': target_air_obstacle_mask,
            'ground_obstacle_mask_img': target_ground_obstacle_mask_img,
            'ground_obstacle_mask': target_ground_obstacle_mask,
        }

    def get_local_perturbation_for_img_info(self, img_info):
        '''Get a list of local perturbations.'''
        up, down, left, right = 0, 0, 0, 0
        if self.scene_idx == 'knife_on_food':
            if img_info.get('knife_above_close_to_food') is not None:
                down = img_info['knife_above_close_to_food'] + 1
                assert down >= 0

        return [up, down, left, right]

    def get_train_data_for_multi_objects_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_idx_to_data_dict[idx]
        else: data_dict = self.test_idx_to_data_dict[idx]

        img_path = data_dict['img_path']

        img = image_utils.load_image_at_path(
                img_path, self.transforms, desired_shape=(256,256))

        # Can be None.
        img_info = data_dict.get('img_info')
        assert img_info is not None, "Cannot get image info for multi-objects"

        if img_info.get('knife_obj_bb_xyxy') is not None:
            knife_obj_bb = [i for i in img_info['knife_obj_bb_xyxy']]
        else:
            knife_obj_bb = None

        if img_info.get('food_obj_bb_xyxy') is not None:
            food_obj_bb = [i for i in img_info['food_obj_bb_xyxy']]
        else:
            food_obj_bb = None

        # Found nothing.
        if food_obj_bb is None and knife_obj_bb is None:
            return None
        else:
            if self.scene_idx == 'knife_on_food':
                if (food_obj_bb is None or knife_obj_bb is None) \
                    and img_info['obstacle_pos_idx'] == 0:
                    return None

        # All obstacles are ground obstacles, air_obstacles were in older
        # versions and this code is kept here only for ??
        # Get bb for air obstacles
        air_obstacle_bb_list, i = [], 0
        while img_info.get('air_obstacle_{}_bb_xyxy'.format(i)) is not None:
            air_obstacle_bb_list.append(
                img_info['air_obstacle_{}_bb_xyxy'.format(i)])
            i = i + 1

        # Get bb for ground obstacles
        ground_obstacle_bb_list, i = [], 0
        while img_info.get('obstacle_{}_bb_xyxy'.format(i)) is not None:
            ground_obstacle_bb_list.append(
                img_info['obstacle_{}_bb_xyxy'.format(i)])
            i = i + 1

        knife_left_food = img_info.get('knife_left_of_food')
        obstacle_pos_idx = img_info.get('obstacle_pos_idx')

        img_tensor = F.to_tensor(img)

        # Get all masked images and the masks
        # Get image (and mask) with knife-food only.
        if knife_obj_bb is not None:
            knife_food_img_tensor, knife_food_mask_tensor = \
                self.get_masked_tensor_with_bb(
                    img_tensor, knife_obj_bb, food_obj_bb)
            # Get image (and mask) with knife-obstacle only.
            knife_obj_masks = self.get_masked_image_tensor_for_target_bb(
                img_tensor, knife_obj_bb, ground_obstacle_bb_list,
                air_obstacle_bb_list)
        else:
            knife_food_img_tensor, knife_food_mask_tensor = None, None
            knife_obj_masks = {}

        if food_obj_bb is not None:
            # Get image (and mask) with food-obstacle only.
            food_obj_masks = self.get_masked_image_tensor_for_target_bb(
                img_tensor, food_obj_bb, ground_obstacle_bb_list,
                air_obstacle_bb_list)
        else:
            food_obj_masks = {}

        obs_obs_img_tensor_list, obs_obs_mask_tensor_list = [], []
        for obs_i, bb_i in enumerate(ground_obstacle_bb_list):
            obs_i_img_tensor_list, obs_i_mask_tensor_list = [], []
            for obs_j, bb_j in enumerate(ground_obstacle_bb_list):
                if obs_i == obs_j:
                    continue
                obs_pair_img, obs_pair_mask = self.get_masked_tensor_with_bb(
                    img_tensor, bb_i, bb_j)
                obs_i_img_tensor_list.append(obs_pair_img)
                obs_i_mask_tensor_list.append(obs_pair_mask)
            obs_obs_img_tensor_list.append(obs_i_img_tensor_list)
            obs_obs_mask_tensor_list.append(obs_i_mask_tensor_list)

        # Get individual images i.e. images with only one object.
        # Get knife image (and mask).
        if knife_obj_bb is not None:
            knife_mask_img_tensor, _ = self.get_masked_tensor_with_bb(
                img_tensor, knife_obj_bb)

        # Get food image and mask.
        if food_obj_bb is not None:
            food_mask_img_tensor, _ = self.get_masked_tensor_with_bb(
                img_tensor, food_obj_bb)

        # Get obstacle images and masks.
        ground_obstacle_img_tensor = [self.get_masked_tensor_with_bb(
            img_tensor, b)[0] for b in ground_obstacle_bb_list]
        air_obstacle_img_tensor = [self.get_masked_image_with_bb(
            img_tensor, b)[0] for b in air_obstacle_bb_list]

        img_arr = np.asarray(img)
        '''
        if len(obs_obs_img_tensor_list) > 0:
            import matplotlib.pyplot as plt
            plt.imshow(img_tensor.cpu().numpy().transpose(1,2,0))
            plt.show()
            plt.imshow(obs_obs_img_tensor_list[0][0].cpu().numpy().transpose(1,2,0))
            plt.show()
            plt.imshow(obs_obs_mask_tensor_list[0][0].cpu().numpy().transpose(1,2,0))
            plt.show()
            plt.imshow(obs_obs_img_tensor_list[1][0].cpu().numpy().transpose(1,2,0))
            plt.show()
            plt.imshow(obs_obs_mask_tensor_list[1][0].cpu().numpy().transpose(1,2,0))
            plt.show()
            plt.imshow(knife_food_mask_tensor.cpu().numpy().transpose(1, 2, 0))
            plt.show()
            for t in range(len(knife_obj_masks['ground_obstacle_mask_img'][0])):
                plt.imshow(knife_obj_masks['ground_obstacle_mask_img'][t].cpu().numpy().transpose(1,2,0))
                plt.show()
                plt.imshow(knife_obj_masks['ground_obstacle_mask'][t].cpu().numpy().transpose(1,2,0))
                plt.show()
                plt.imshow(food_obj_masks['ground_obstacle_mask_img'][t].cpu().numpy().transpose(1, 2, 0)),
                plt.show()
                plt.imshow(food_obj_masks['ground_obstacle_mask'][t].cpu().numpy().transpose(1, 2, 0)),
                plt.show()
            import ipdb; ipdb.set_trace()
        '''

        has_obstacle_between_knife_food = \
            img_info.get('has_obstacle_between_knife_food')
        if self.scene_idx == 'knife_on_food':
            has_obstacle_between_knife_food = 0 \
                if img_info['knife_pos_idx'] == 0 else 1
            knife_above_close_to_food = img_info.get(
                'knife_above_close_to_food', -1)

        local_perturb_info = self.get_local_perturbation_for_img_info(img_info)
        result_dict = {
            'info': {
                'img_path': img_path,
                'has_obstacle_bw_knife_food': has_obstacle_between_knife_food,
                'knife_left_food': knife_left_food,
                'obstacle_pos_idx': obstacle_pos_idx,
                'knife_food_in_air': img_info.get('knife_food_in_air'),
                'local_perturb_info': local_perturb_info,
            },
            'bounding_box_info': {
                'knife_obj': knife_obj_bb,
                'food_obj': food_obj_bb,
                'ground_obj_bb_list': ground_obstacle_bb_list,
                'air_obj_bb_list': air_obstacle_bb_list,
            },
            'img': img_tensor,
            'knife_food_mask_img': knife_food_img_tensor,
            'knife_obstacle_air_mask_img':
                knife_obj_masks.get('air_obstacle_mask_img'),
            'knife_obstacle_ground_mask_img':
                knife_obj_masks.get('ground_obstacle_mask_img'),
            'food_obstacle_air_mask_img':
                food_obj_masks.get('air_obstacle_mask_img'),
            'food_obstacle_ground_mask_img':
                food_obj_masks.get('ground_obstacle_mask_img'),
            'obstacle_obstacle_mask_img': obs_obs_img_tensor_list,

            'obj_img': {
                'knife_img': knife_mask_img_tensor,
                'food_img': food_mask_img_tensor,
                'ground_obstacle_img': ground_obstacle_img_tensor,
                'air_obstacle_img': air_obstacle_img_tensor,
            }
        }

        if self.load_bb_masks_for_relations:
            result_dict['obj_pair_img_mask_with_bb'] = {
                'knife_food': knife_food_mask_tensor,
                'knife_obstacle_air': knife_obj_masks['air_obstacle_mask'],
                'knife_obstacle_ground':
                    knife_obj_masks['ground_obstacle_mask'],
                'food_obstacle_air': food_obj_masks['air_obstacle_mask'],
                'food_obstacle_ground': food_obj_masks['ground_obstacle_mask'],
                'obstacle_obstacle': obs_obs_mask_tensor_list,
            }

        return result_dict

    def get_all_bounding_box_from_img_info(self, img_info):
        steps_before = img_info['action']['steps_before_action']
        steps_after = img_info['action']['steps_after_action']
        total_steps = steps_before + steps_after
        anchor_bb_before = img_info['{}_anchor_obj_bb_xyxy'.format(steps_before)]
        other_bb_before = img_info['{}_other_obj_bb_xyxy'.format(steps_before)]

        anchor_bb_after = img_info['{}_anchor_obj_bb_xyxy'.format(total_steps)]
        other_bb_after = img_info['{}_other_obj_bb_xyxy'.format(total_steps)]

        # TODO(Mohit): Add some regularization here
        delta_other_bb = [other_bb_after[i] - other_bb_before[i]
                          for i in range(4)]

        # anchor_bb_before = [i/256.0 for i in anchor_bb_before]
        # anchor_bb_after = [i/256.0 for i in anchor_bb_after]
        # other_bb_before = [i/256.0 for i in other_bb_before]
        # other_bb_after = [i/256.0 for i in other_bb_after]
        # delta_other_bb = [i/256.0 for i in delta_other_bb]

        return {
            'anchor_bb_before': anchor_bb_before,
            'other_bb_before': other_bb_before,
            'anchor_bb_after': anchor_bb_after,
            'other_bb_after': other_bb_after,
            'other_bb_delta': delta_other_bb,
        }
    
    def remove_white_color_from_img(self, img):
        white_color = [255, 255, 255]
        mask = np.all(img == white_color, axis=-1)
        img[mask] = [0, 0, 0]
        return img

    def get_train_data_at_idx(self, idx, train=True):
        if train:
            data_dict = self.train_idx_to_data_dict[idx]
        else:
            data_dict = self.test_idx_to_data_dict[idx]

        img_path = data_dict['img_path']
        img = image_utils.load_image_at_path(
                img_path, self.transforms, desired_shape=(256, 256))
        img_arr = np.array(img)
        img = self.remove_white_color_from_img(img_arr)

        img_info = data_dict.get('img_info')
        all_bb_info = self.get_all_bounding_box_from_img_info(img_info)
        img_tensor = F.to_tensor(img)

        data = {
            'img': img_tensor,
            'img_path': img_path,
            'img_info': img_info,
            'bb_info': all_bb_info,
            'action_info': data_dict['img_info']['action'],
        }
        return data
