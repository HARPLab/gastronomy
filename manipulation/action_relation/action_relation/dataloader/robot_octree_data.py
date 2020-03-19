import numpy as np 

import os
import pickle 
import open3d as o3d
from collections import OrderedDict

from utils import image_utils
from utils.transformations import rotation_matrix
from action_relation.utils.open3d_utils import read_point_cloud, make_pcd

import typing


def convert_voxel_index_to_3d_index(xyz_arr, min_xyz, xyz_size, voxel_size, 
                                    validate=True):
    idx = (xyz_arr - min_xyz) / voxel_size
    idx_int = np.around(idx, decimals=1).astype(np.int32)
    # idx_int = idx.astype(np.int32)
    if validate:
        for i in range(3):
            if type(idx_int) is np.ndarray and len(idx_int.shape) > 1:
                assert np.all(idx_int >= 0) and np.all(idx_int[:, i] <= xyz_size[i])
            else:
                assert idx_int[i] >= 0 and idx_int[i] <= xyz_size[i]
    return idx_int


class SceneVoxels(object):
    '''Create a single 3D representation for the entire scene. This is used
    for training a precond classifier directly from scene representation.
    
    What happens when the entire scene does not fit in the voxel space?
    '''
    def __init__(self, pcd_path_list, scene_type):
        self.pcd_path_list = pcd_path_list
        self.scene_type = scene_type

        self.voxel_size: float = 0.01
        self.min_xyz = np.array([-0.5, -0.5, -0.25])
        self.max_xyz = np.array([0.5, 0.5, 0.5])

        self.xyz_size = np.around(
            (self.max_xyz-self.min_xyz)/self.voxel_size, decimals=1)
        self.xyz_size = self.xyz_size.astype(np.int32)

        self.full_3d = None
        self.voxel_index_int = None
        self.save_full_3d = True

        pcd_list, pcd_points_arr_list = [], []
        min_z_per_pcd_list = []
        for pcd_path in pcd_path_list:
            pcd = read_point_cloud(pcd_path)
            pcd_list.append(pcd)

            pcd_points_arr = np.asarray(pcd.points)
            pcd_points_arr_list.append(pcd_points_arr)

            [_, _, min_z] = pcd_points_arr.min(axis=0)
            min_z_per_pcd_list.append(min_z)
        
        min_z_per_pcd_list = sorted(min_z_per_pcd_list)

        if scene_type == "data_in_line":
            # Make sure that the objects are in a plane ?
            assert min_z_per_pcd_list[1] - min_z_per_pcd_list[0] <= 0.01
        elif scene_type == "cut_food":
            pass
        elif scene_type == "box_stacking":
            pass
        else:
            raise ValueError(f"Invalid scene type {scene_type}")

        new_world_origin = [0, 0, min_z]
        x_axis, rot_angle = [1, 0, 0], np.deg2rad(180.0)
        self.T = rotation_matrix(rot_angle, x_axis)
        self.T[:3, 3] = new_world_origin

        self.pcd_points_arr_list = []
        for pcd in pcd_list:
            pcd.transform(self.T)
            pcd_arr = np.asarray(pcd.points)
            self.pcd_points_arr_list.append(pcd_arr)

    def init_voxel_index(self) -> bool:
        # The 
        self.object_voxel_index_int_list = []
        for pcd_points in self.pcd_points_arr_list:
            pcd_idx = (pcd_points- self.min_xyz) / self.voxel_size
            pcd_index_int = np.around(pcd_idx, decimals=1).astype(np.int32)

            if np.any(pcd_index_int.max(axis=0) >= self.xyz_size):
                print("==== ERROR: Object out of bounds (above max) ====")
                return False

            if np.any(pcd_index_int.min(axis=0) < 0):
                print("==== ERROR: Object out of bounds (below min) ====")
                return False
            
            self.object_voxel_index_int_list.append(pcd_index_int)

        if self.save_full_3d:
            self.status_3d, self.full_3d = self.parse()     

            # Remove the other data to make up space
            self.anchor_index_int = None
            self.other_index_int = None
            self.other_obj_voxels_idx = None

        return True

    def convert_voxel_index_to_3d_index(self, xyz_arr, validate=True):
        return convert_voxel_index_to_3d_index(
            xyz_arr, self.min_xyz, self.xyz_size, self.voxel_size)

    def parse(self):
        # if self.save_full_3d and self.objects_are_far_apart:
        #     return True, None

        if self.save_full_3d and self.full_3d is not None:
            return self.status_3d, self.full_3d
        
        if self.full_3d is None:
            full_3d =  np.zeros([2] + self.xyz_size.tolist())
            object_idx = 1
            for object_index_int in self.object_voxel_index_int_list:
                full_3d[0,
                        object_index_int[:, 0],
                        object_index_int[:, 1],
                        object_index_int[:, 2]] = 1
                full_3d[1,
                        object_index_int[:, 0],
                        object_index_int[:, 1],
                        object_index_int[:, 2]] = object_idx
                object_idx += 1

        return True, full_3d
    
    def convert_full3d_arr_to_open3d(self) -> dict:
        status, full_3d = self.parse()

        if full_3d is None:
            return {}

        ax_x, ax_y, ax_z = np.where(full_3d[0, ...] != 0)
        ax = np.vstack([ax_x, ax_y, ax_z]).T
        scene_pcd = make_pcd(ax, color=[1, 0, 0])

        return {'scene': scene_pcd}
    
    def visualize_full3d(self) -> None:
        status, voxels_arr = self.parse()

        if voxels_arr is None:
            return
        
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = voxels_arr[0, ...].nonzero()
        ax.scatter(x, y, z)
        # ax.set_title('', fontsize=14)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
    

class RobotAllPairVoxels(object):
    def __init__(self, pcd_path_list):
        self.pcd_path_list = pcd_path_list

        self.voxel_size: float = 0.01
        self.min_xyz = np.array([-0.25, -0.25, -0.25])
        self.max_xyz = np.array([0.25, 0.25, 0.25])

        self.xyz_size = np.around(
            (self.max_xyz-self.min_xyz)/self.voxel_size, decimals=1)
        self.xyz_size = self.xyz_size.astype(np.int32)

        self.pcd_list = [
            read_point_cloud(pcd_path) for pcd_path in pcd_path_list
        ]

        self.pcd_list, self.pcd_points_arr_list = [], []
        self.min_z_per_pcd_list = []
        self.obj_center_per_pcd_list = []
        for pcd_path in pcd_path_list:
            pcd = read_point_cloud(pcd_path)
            self.pcd_list.append(pcd)

            pcd_points_arr = np.asarray(pcd.points)
            self.pcd_points_arr_list.append(pcd_points_arr)

            [_, _, min_z] = pcd_points_arr.min(axis=0)
            [mean_x, mean_y, mean_z] = pcd_points_arr.mean(axis=0)
            self.min_z_per_pcd_list.append(min_z)
            self.obj_center_per_pcd_list.append([mean_x, mean_y, mean_z])
        
        self.min_z_per_pcd_list = sorted(self.min_z_per_pcd_list)
        # Make sure that the objects are in a plane ?
        # assert self.min_z_per_pcd_list[1] - self.min_z_per_pcd_list[0] <= 0.01

        self.robot_voxels_by_pcd_pair_dict = OrderedDict()

    def init_voxels_for_pcd_pair(self, anchor_idx, other_idx):
        robot_voxels = RobotVoxels(self.pcd_list[anchor_idx], 
                                   self.pcd_list[other_idx],
                                   min_xyz=self.min_xyz,
                                   max_xyz=self.max_xyz)
        status = robot_voxels.init_voxel_index()
        self.robot_voxels_by_pcd_pair_dict[(anchor_idx, other_idx)] = robot_voxels

        anchor_center = np.array(self.obj_center_per_pcd_list[anchor_idx])
        other_center = np.array(self.obj_center_per_pcd_list[other_idx])
        dist = np.linalg.norm(anchor_center - other_center)
        print(f"inter obj dist: {dist}")

        return status, robot_voxels
    
    def get_object_center_list(self):
        return self.obj_center_per_pcd_list


class RobotVoxels(object):
    def __init__(self, 
                 anchor_pcd: o3d.geometry.PointCloud, 
                 other_pcd: o3d.geometry.PointCloud, 
                 min_xyz,
                 max_xyz,
                 has_object_in_between=False) -> None:

        self.anchor_pcd = o3d.geometry.PointCloud(anchor_pcd)
        self.other_pcd = o3d.geometry.PointCloud(other_pcd)

        self.voxel_size: float = 0.01
        self.has_object_in_between = has_object_in_between
        self.objects_are_far_apart = False

        # This should be similar to simulation or atleast the final data 
        # size should be similar to simulation.
        # self.min_xyz = np.array([-0.65, -0.65, -0.5])
        # self.max_xyz = np.array([0.65, 0.65, 0.5])

        self.min_xyz = min_xyz
        self.max_xyz = max_xyz

        self.xyz_size = np.around(
            (self.max_xyz-self.min_xyz)/self.voxel_size, decimals=1)
        self.xyz_size = self.xyz_size.astype(np.int32)
        
        self.full_3d = None
        self.voxel_index_int = None
        self.save_full_3d = True

        # The original world coordinate system is X to right, Y ahead and Z down
        # We want to move the origin to the base of the anchor object with X ahead
        # Y to right and Z up.

        anchor_points = np.asarray(self.anchor_pcd.points)
        other_points = np.asarray(self.other_pcd.points)

        [center_x, center_y, _] = anchor_points.mean(axis=0)
        [_, _, min_z] = anchor_points.min(axis=0) 
        new_world_origin = [-center_x, -center_y, -min_z]

        x_axis, rot_angle = [1, 0, 0], np.deg2rad(0)
        self.T = rotation_matrix(rot_angle, x_axis)
        self.T[:3, 3] = new_world_origin

        self.anchor_pcd.transform(self.T)
        self.other_pcd.transform(self.T)

        self.T2 = rotation_matrix(np.deg2rad(180), x_axis)
        self.anchor_pcd.transform(self.T2)
        self.other_pcd.transform(self.T2)

        self.transf_anchor_points = np.asarray(self.anchor_pcd.points)
        self.transf_other_points = np.asarray(self.other_pcd.points)

    
    def init_voxel_index(self) -> bool:
        # The 
        anchor_idx = (self.transf_anchor_points - self.min_xyz) / self.voxel_size
        self.anchor_index_int = np.around(anchor_idx, decimals=1).astype(np.int32)
        other_idx = (self.transf_other_points - self.min_xyz) / self.voxel_size
        self.other_index_int = np.around(other_idx, decimals=1).astype(np.int32)

        if np.any(self.other_index_int.max(axis=0) >= self.xyz_size):
            #  The other object is too far.
            self.objects_are_far_apart = True
            # raise ValueError("other Object is out of boundary")
        
        if np.any(self.other_index_int.min(axis=0) < 0):
            #  The other object is too far.
            self.objects_are_far_apart = True
            # raise ValueError("other Object is out of boundary")
        
        # print("Objects are far apart: {}, have obstacle in between: {}".format(
        #     self.objects_are_far_apart, self.has_object_in_between
        # ))

        if self.objects_are_far_apart:
            return True
        
        if self.save_full_3d:
            self.status_3d, self.full_3d = self.parse()     

            # Remove the other data to make up space
            self.anchor_index_int = None
            self.other_index_int = None
            self.other_obj_voxels_idx = None

        return True

    def convert_voxel_index_to_3d_index(self, xyz_arr, validate=True):
        return convert_voxel_index_to_3d_index(
            xyz_arr, self.min_xyz, self.xyz_size, self.voxel_size)

    def create_position_grid(self):
        grid = np.meshgrid(np.arange(self.xyz_size[0]),
                           np.arange(self.xyz_size[1]),
                           np.arange(self.xyz_size[2]))
        voxel_0_idx = self.convert_voxel_index_to_3d_index(np.zeros(3))
        grid[0] = grid[0] - voxel_0_idx[0]
        grid[1] = grid[1] - voxel_0_idx[1]
        grid[2] = grid[2] - voxel_0_idx[2]
        return np.stack(grid)
    
    def get_all_zero_voxels(self):
        '''Returns canonical voxles which are all 0's.'''
        full_3d =  np.zeros([3] + self.xyz_size.tolist())
        return full_3d
    
    def parse(self):
        if self.save_full_3d and self.objects_are_far_apart:
            return True, None

        if self.save_full_3d and self.full_3d is not None:
            return self.status_3d, self.full_3d
        
        if self.full_3d is None:
            full_3d =  np.zeros([3] + self.xyz_size.tolist())
            full_3d[0,
                    self.anchor_index_int[:, 0],
                    self.anchor_index_int[:, 1],
                    self.anchor_index_int[:, 2]] = 1
            full_3d[0,
                    self.other_index_int[:, 0], 
                    self.other_index_int[:, 1],
                    self.other_index_int[:, 2]] = 2
            full_3d[1,
                    self.anchor_index_int[:, 0],
                    self.anchor_index_int[:, 1],
                    self.anchor_index_int[:, 2]] = 1
            full_3d[2,
                    self.other_index_int[:, 0], 
                    self.other_index_int[:, 1],
                    self.other_index_int[:, 2]] = 1

        return True, full_3d
    
    def convert_full3d_arr_to_open3d(self) -> dict:
        status, full_3d = self.parse()

        if full_3d is None:
            return {}

        ax_x, ax_y, ax_z = np.where(full_3d[1, ...] != 0)
        ax = np.vstack([ax_x, ax_y, ax_z]).T
        anchor_pcd = make_pcd(ax, color=[1, 0, 0])

        ax_x, ax_y, ax_z = np.where(full_3d[2, ...] != 0)
        ax = np.vstack([ax_x, ax_y, ax_z]).T
        other_pcd = make_pcd(ax, color=[0, 0, 1])

        return {'anchor': anchor_pcd, 'other': other_pcd}
    
    def visualize_full3d(self) -> None:
        status, voxels_arr = self.parse()

        if voxels_arr is None:
            return
        
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = voxels_arr[0, ...].nonzero()
        ax.scatter(x, y, z)
        # ax.set_title('', fontsize=14)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()


def create_robot_voxels_from_anchor_pcd_path(
    anchor_pcd_path: str, 
    other_pcd_path: str,
    has_object_in_between: bool) -> RobotVoxels:

    anchor_pcd = read_point_cloud(anchor_pcd_path)
    other_pcd = read_point_cloud(other_pcd_path)

    return RobotVoxels(anchor_pcd, other_pcd, has_object_in_between)
