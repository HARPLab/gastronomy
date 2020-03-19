import numpy as np

from utils.transformations import *

import pdb

def _convert_points_to_homogenous_form(point_arr):
    assert len(point_arr.shape) == 2 and point_arr.shape[1] in (2, 3)
    return np.hstack([point_arr, np.ones(point_arr.shape[0]).reshape(-1, 1)])


def get_index_for_action(action):
    action_i = []
    for i in range(8):
        _f = get_filter_for_push_action(i)
        action_i.append(_f(action))
    assert np.sum(action_i) == 1
    return np.where(action_i)[0][0]

def get_index_for_push_action(action):
    action_i = []
    for i in range(8):
        _f = get_filter_for_push_action(i)
        action_i.append(_f(action))
    assert np.sum(action_i) == 1
    return np.where(action_i)[0][0]

def get_filter_for_push_action(action_idx):
    '''Returns a function which accepts an action and filters similar actions.
    
    action_idx: Actions are generated using (cos(theta), sin(theta), 0).
    '''
    pi_4 = np.pi / 4.0
    action_list = [[np.cos(t*pi_4), np.sin(t*pi_4), 0] for t in range(0, 8)]
    assert action_idx < len(action_list)
    main_action = action_list[action_idx]
    def _f(action):
        if type(action) is list:
            action = np.array(action)
        assert type(action) is np.ndarray, "Invalid action type"
        action_norm = action / np.linalg.norm(action)
        for i in range(len(main_action)):
            if not abs(main_action[i] - action_norm[i]) < 1e-3:
                return False

        return True
    
    return _f


def get_filter_for_action(action_idx):
    '''Returns a function which accepts an action and filters similar actions.
    
    action_idx: Actions are generated using (cos(theta), sin(theta), 0).
    '''
    pi_4 = np.pi / 4.0
    # Actions for ground data
    # action_list = [[np.cos(t*pi_4), np.sin(t*pi_4), 0] for t in range(0, 8)]
    # Actions for robot vrep data
    action_list = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], 
                   [0, 0, -1], [0, 0, 1]]
    assert action_idx < len(action_list)
    main_action = action_list[action_idx]
    def _f(action):
        if type(action) is list:
            action = np.array(action)
        assert type(action) is np.ndarray, "Invalid action type"
        action_norm = action / np.linalg.norm(action)
        for i in range(len(main_action)):
            if not abs(main_action[i] - action_norm[i]) < 1e-3:
                return False

        return True
    
    return _f


class VrepSceneGeometry(object):

    def __init__(self, geom_data, path, parse_before_scene_only=False):
        self.geom_data = geom_data
        self.path = path
        self.parse_before_scene_only = parse_before_scene_only
        self.relative_action_info = None

        self.voxel_disp_type = 'discrete_real_disp'

        self.parse_scene()
    
    def parse_scene(self):
        self.before_geom = self.get_data_from_dict(self.geom_data['before'])
        if not self.parse_before_scene_only:
            self.after_geom = self.get_data_from_dict(self.geom_data['after'])

    def transform_bounding_box(self, bb, T, return_homogenous=False):
        bb_transf = np.dot(T, bb.T)
        # min_p = np.min(bb_transf, axis=1)
        # max_p = np.max(bb_transf, axis=1)
        min_p = bb_transf[:, 0]
        max_p = bb_transf[:, 1]

        if return_homogenous:
            return min_p, max_p
        else:
            return min_p[:3], max_p[:3]
    
    def get_position_for_object(self, before=True, anchor_obj=True):
        geom_dict = self.before_geom if before else self.after_geom
        if anchor_obj:
            p = geom_dict['anchor_pos']
        else:
            p = geom_dict['other_pos']
        return p

    def get_bounding_box_in_object_frame(self, before=True, anchor_obj=True):
        geom_dict = self.before_geom if before else self.after_geom
        if anchor_obj:
            bb = geom_dict['anchor_bb']
        else:
            bb = geom_dict['other_bb']
        return bb
    
    def get_transformation_for_object(self, before=True, anchor_obj=True):
        '''Returns a (3, 4) transformation matrix.'''
        geom_dict = self.before_geom if before else self.after_geom
        if anchor_obj:
            T = geom_dict['anchor_T']
        else:
            T = geom_dict['other_T']
        return T

    def get_other_oriented_bounding_box(self, before=True):
        geom_dict = self.before_geom if before else self.after_geom
        # bb in object frame
        bb = geom_dict['other_bb']
        T = geom_dict['other_T']
        bb_homo = _convert_points_to_homogenous_form(bb)
        # get bb in world frame.
        min_p, max_p = self.transform_bounding_box(bb_homo, T)
        return min_p, max_p

    def get_anchor_oriented_bounding_box(self, before=True):
        geom_dict = self.before_geom if before else self.after_geom
        # bb in object frame
        bb = geom_dict['anchor_bb']
        T = geom_dict['anchor_T']
        bb_homo = _convert_points_to_homogenous_form(bb)
        # get bb in world frame.
        min_p, max_p = self.transform_bounding_box(bb_homo, T)
        return min_p, max_p
    
    def get_real_by_max_disp_ratio(self, multiply_by_relative_action=True):
        '''Get the ratio of real/max displacement to categorize similar scenes
        together. This is only useful for relative actions since for absolute
        actions the denominator would always be the same thus allowing us to use
        the real displacement directly.
        '''
        anchor_pos_before = self.get_position_for_object(before=True, anchor_obj=True)
        other_pos_before = self.get_position_for_object(before=True, anchor_obj=False)
        other_pos_after = self.get_position_for_object(before=False, anchor_obj=False)

        # Take the absolute of max displacement so that the action affect is 
        # only used and not the direction?
        # max_disp = np.abs(anchor_pos_before - other_pos_before)
        max_disp = anchor_pos_before - other_pos_before
        real_disp = other_pos_after - other_pos_before
        real_by_max_disp_voxel = np.abs(real_disp) / (np.abs(max_disp) + 1e-6)
        if multiply_by_relative_action:
            action_arr = np.abs(self.relative_action_info['sgn'])
            assert np.sum(action_arr) == 1
            real_by_max_disp_voxel = real_by_max_disp_voxel * action_arr
        return real_by_max_disp_voxel

    
    def get_voxel_displacement(self):
        anchor_pos_before = self.get_position_for_object(before=True, anchor_obj=True)
        other_pos_before = self.get_position_for_object(before=True, anchor_obj=False)
        other_pos_after = self.get_position_for_object(before=False, anchor_obj=False)

        max_disp = anchor_pos_before - other_pos_before
        real_disp = other_pos_after - other_pos_before
        
        if self.voxel_disp_type == 'discrete_real_disp':
            voxel_disp = np.around(real_disp / 0.01)
        else:
            raise ValueError("Invalid voxel disp.")

        return voxel_disp
    
    def get_action_data_for_relative_action(self, use_true_action_idx=False):
        ''' Return a tuple of action idx and action as one hot value.
        '''
        if use_true_action_idx:
            idx_key, one_hot_key = 'true_action_idx', 'true_action_idx_one_hot'
        else:
            idx_key, one_hot_key = 'action_idx', 'action_idx_one_hot'
        assert np.sum(self.relative_action_info[one_hot_key]) == 1
        idx = self.relative_action_info[idx_key]
        assert self.relative_action_info[one_hot_key][idx] == 1
        return self.relative_action_info[idx_key], self.relative_action_info[one_hot_key]

    def get_and_save_info_for_relative_action(self):
        if self.relative_action_info is None:
            action = self.geom_data['action']
            self.relative_action_info = self.get_info_for_relative_action(action)
        return self.relative_action_info    
    
    def get_max_possible_displacement(self):
        anchor_pos_before = self.get_position_for_object(before=True, anchor_obj=True)
        other_pos_before = self.get_position_for_object(before=True, anchor_obj=False)
        return anchor_pos_before - other_pos_before
    
    def get_real_displacement(self):
        other_pos_before = self.get_position_for_object(before=True, anchor_obj=False)
        other_pos_after = self.get_position_for_object(before=False, anchor_obj=False)
        return other_pos_after - other_pos_before

    def get_info_for_relative_action(self, action):
        debug = False
        anchor_pos_before = self.get_position_for_object(before=True, anchor_obj=True)
        other_pos_before = self.get_position_for_object(before=True, anchor_obj=False)
        other_pos_after = self.get_position_for_object(before=False, anchor_obj=False)

        max_disp = anchor_pos_before - other_pos_before
        real_disp = other_pos_after - other_pos_before
        total_disp = np.sum(np.abs(real_disp))

        # The sum will be 1 if we have motion in one direction but it can also 
        # be zero if the object cannot move in that direction.
        assert np.sum(np.abs(real_disp) > 0.02) <= 1, "Displacement in multiple directions"
        
        # Make sure that we have real_disp <= max_disp 
        for i in range(3):
            if abs(max_disp[i]) + 0.001 < abs(real_disp[i]):
                if debug:
                    print(f"Occurred disp diff: {abs(max_disp[i] - real_disp[i]):.5f} "
                        f"larger than possible: {self.path}")
                # import pdb; pdb.set_trace()
                return None

        # We should use a ratio
        max_disp_voxel = [max_disp[i] // 0.01 for i in range(3)]
        real_disp_voxel = [real_disp[i] // 0.01 for i in range(3)]
        real_by_max_disp_voxel = [(real_disp[i] / max_disp[i]) // 0.01
                                   for i in range(3)]

        sign_disp = []
        for i in range(3):
            if abs(real_disp[i]) <= 0.002:
                sign = 0
                if action[i] == 0: 
                    # Great this is not motion that we desired
                    pass
                else:
                    # Ok this is motion that we desired so maybe the anchor is 
                    # blocking`
                    assert total_disp - abs(real_disp[i]) <= 0.002
                    sign = 1

            elif real_disp[i] * max_disp[i] > 0:
                sign = 1
                if action[i] != 1:
                    if debug:
                        print(f"ERROR?: motion in direction {i} but gt action: {action[i]}")
                    sign = -1
            else:
                sign = -1
                if action[i] == 0:
                    if debug:
                        print(f"ERROR?: No motion in direction {i} yet there is "
                            f"disp {real_disp[i]}")
                    sign = 0
                # assert action[i] != 0
            sign_disp.append(sign)
        
        # Get the real action i.e. moving left or right. During simulation we 
        # take action based on offsets i.e., we either move [+1, 0, 0] or [-1, 0, 0]
        # and then take diff = anchor_pos - other_pos and move to other_pos + diff * action.
        # Thus these actions do not have a fixed direction, their direction is relative to
        # where the anchor is. Hence here we find the true_action_idx
        # true_action_idx_0_1 = 1 if np.sum((max_disp > 0.0) * action) > 0 else 0
        true_action_idx_0_1 = 1 if np.sum(np.sign(max_disp) * action) > 0 else 0

        # if '00040' in self.path and 'sample_2_edge_1_out' in self.path:
        #     pdb.set_trace()
        
        # Convert sign to index
        sign_disp_arr = np.int32(sign_disp)
        if sign_disp_arr.sum() == -1:
            action_idx = 2 * sign_disp_arr.nonzero()[0][0]
        else:
            action_idx = 2 * sign_disp_arr.nonzero()[0][0] + 1
        true_action_idx = 2 * sign_disp_arr.nonzero()[0][0] + true_action_idx_0_1
        assert action_idx >= 0 and action_idx < 6, "Invalid action idx"
        assert true_action_idx >= 0 and true_action_idx < 6, "Invalid true action idx"

        action_idx_one_hot = [0] * 6
        action_idx_one_hot[action_idx] = 1
        true_action_idx_one_hot = [0] * 6
        true_action_idx_one_hot[true_action_idx] = 1

        result_dict = dict(
            action_idx=action_idx,
            action_idx_one_hot=action_idx_one_hot,
            true_action_idx=true_action_idx,
            true_action_idx_one_hot=true_action_idx_one_hot,
            sgn=sign_disp,
            max_disp_voxel=max_disp_voxel,
            real_disp_voxel=real_disp_voxel,
            real_by_max_disp_voxel=real_by_max_disp_voxel,
        )
        return result_dict
        

    def get_data_from_dict(self, d):
        return dict(
            anchor_pos=np.array(d['anchor_pos']),
            anchor_angles=np.array(d['anchor_q']),
            anchor_bb=np.array(d['anchor_bb']).reshape(-1, 3),
            anchor_T=np.array(d['anchor_T_matrix'] + [0, 0, 0, 1]).reshape(-1, 4),

            other_pos=np.array(d['other_pos']),
            other_angles=np.array(d['other_q']),
            other_bb=np.array(d['other_bb']).reshape(-1, 3),
            other_T=np.array(d['other_T_matrix'] + [0, 0, 0, 1]).reshape(-1, 4),
        )

    def check_if_same_other_obj_z_axis_before_after(self):
        '''Checks the z-axis for the other object before and after action.
        
        Return: 0 if z-axis is the same before and after else 1.
        '''
        z_before = self.before_geom['other_T'][2][:3]
        z_after = self.after_geom['other_T'][2][:3]
        assert abs(np.linalg.norm(z_before) - 1) < 1e-4
        assert abs(np.linalg.norm(z_after) - 1) < 1e-4

        if np.all((z_before - z_after) < 1e-2):
            # the orientation does not change but it could still be oriented
            # interestingly.
            # z_anchor = self.before_geom['anchor_T'][2][:3]
            # val = abs(np.dot(z_anchor, z_before))
            # if val >= 0.01 and val <= 0.98:
                # return False

            # Same 
            return True

        return False

    def create_dense_voxels_from_sparse(self, obj_voxel_arr, before=True, 
                                        anchor_obj=True):
        if before and anchor_obj:
            obj_transform = self.before_geom['anchor_T']        
        elif before and not anchor_obj:
            obj_transform = self.before_geom['other_T']        
        elif not before and not anchor_obj:
            obj_transform = self.after_geom['other_T']        
        else:
            raise ValueError("Invalid option")

        # Take inverse of transform to convert everything in the object space. 
        # This should be axis aligned.
        T_inv = np.linalg.inv(obj_transform)
        sparse_voxels = np.hstack([obj_voxel_arr, 
                                np.ones((obj_voxel_arr.shape[0], 1))])
        transf_sparse_voxels = np.dot(T_inv, sparse_voxels.T)
        transf_sparse_voxels = transf_sparse_voxels.T
        [min_x, min_y, min_z, _] = np.min(transf_sparse_voxels, axis=0)
        [max_x, max_y, max_z, _] = np.max(transf_sparse_voxels, axis=0)
        curr_z, step_z = min_z, 0.01
        dense_obj_voxel_points = []

        while curr_z < max_x:
            points_above_curr_z_idx = transf_sparse_voxels[:, 2] >= curr_z
            points_above_curr_z = transf_sparse_voxels[points_above_curr_z_idx, :3]
            # Project the points onto z = curr_z axis.
            proj_points = np.copy(points_above_curr_z)
            proj_points[:, 2] = curr_z

            dense_obj_voxel_points += proj_points.tolist()
            curr_z += step_z

        dense_obj_voxels_arr = np.array(dense_obj_voxel_points)
        dense_obj_voxels_arr = np.hstack([
            dense_obj_voxels_arr, np.ones((dense_obj_voxels_arr.shape[0], 1))])
        # Transform the points back into the original space
        org_dense_obj_voxels_arr = np.dot(obj_transform, dense_obj_voxels_arr.T)
        org_dense_obj_voxels_arr = (org_dense_obj_voxels_arr[:3, :]).T

        # ==== Visualize ====
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(sparse_voxels[:, 0], sparse_voxels[:, 1], sparse_voxels[:, 2])

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(dense_obj_voxels_arr[:, 0], 
        #            dense_obj_voxels_arr[:, 1], 
        #            dense_obj_voxels_arr[:, 2])

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.show()

        return org_dense_obj_voxels_arr
