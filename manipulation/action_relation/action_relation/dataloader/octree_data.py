import numpy as np 

import os
import pickle 

from utils import image_utils

# Initial min_xyz, max_xyz list
# min_xyz = [-0.65, -0.65, -0.5], [0.65, 0.65, 0.5] if voxel size is 0.025
MIN_XYZ = np.array([-0.25, -0.25, -0.25])
MAX_XYZ = np.array([0.25, 0.25, 0.25])

class OctreeVoxels(object):
    def __init__(self, voxels_list=None, min_xyz=MIN_XYZ, max_xyz=MAX_XYZ):
        if voxels_list is None:
            self.voxels_arr = None
        else:
            self.voxels_arr = np.array(voxels_list).reshape(-1, 3)
        self.voxel_size = 0.01

        # self.min_xyz = np.array([-0.65, -0.65, -0.5])
        # self.max_xyz = np.array([0.65, 0.65, 0.5])

        # For objects with reduced size we should use a smaller volume to capture relations.
        self.min_xyz = min_xyz
        self.max_xyz = max_xyz

        self.xyz_size = np.around(
            (self.max_xyz-self.min_xyz)/self.voxel_size, decimals=1)
        self.xyz_size = self.xyz_size.astype(np.int32)
        # self.xyz_size += 1
        self.full_3d = None
        self.voxel_index_int = None

    def init_voxel_index(self):
        self.voxel_index_int = np.zeros_like(self.voxels_arr).astype(np.int32)
        for i in range(self.voxels_arr.shape[0]):
            idx = self.convert_voxel_index_to_3d_index(self.voxels_arr[i])
            for j in range(len(idx)):
                if self.xyz_size[j] <= idx[j]:
                    return False
            self.voxel_index_int[i] = idx
        return True
    
    def create_position_grid(self):
        grid = np.meshgrid(np.arange(self.xyz_size[0]),
                           np.arange(self.xyz_size[1]),
                           np.arange(self.xyz_size[2]))
        voxel_0_idx = self.convert_voxel_index_to_3d_index(np.zeros(3))
        grid[0] = grid[0] - voxel_0_idx[0]
        grid[1] = grid[1] - voxel_0_idx[1]
        grid[2] = grid[2] - voxel_0_idx[2]
        return np.stack(grid)

    def convert_voxel_index_to_3d_index(self, xyz_arr, validate=True):
        idx = (xyz_arr - self.min_xyz) / self.voxel_size
        idx_int = np.around(idx, decimals=1).astype(np.int32)
        # idx_int = idx.astype(np.int32)
        if validate:
            for i in range(3):
                if type(idx_int) is np.ndarray and len(idx_int.shape) > 1:
                    assert np.all(idx_int >= 0) and np.all(idx_int[:, i] <= self.xyz_size[i])
                else:
                    assert idx_int[i] >= 0 and idx_int[i] <= self.xyz_size[i]
        return idx_int
    
    def parse(self):
        '''Valid octree that fits into the above memory.'''
        if self.full_3d is None:
            # start = time.time()
            full_3d = np.zeros(self.xyz_size)
            full_3d[self.voxel_index_int[:, 0], 
                    self.voxel_index_int[:, 1],
                    self.voxel_index_int[:, 2]] = 1
            # end = time.time()
            # print("Time to load: {:.4f}".format(end - start))

            # for i in range(self.voxels_arr.shape[0]):
                # idx = self.convert_voxel_index_to_3d_index(self.voxels_arr[i])
                # for j in range(len(idx)):
                    # if full_3d.shape[j] <= idx[j]:
                        # return False, None

                # full_3d[idx[0], idx[1], idx[2]] = 1
        
        return True, full_3d

class OctreePickleLoader(OctreeVoxels):
    def __init__(self, voxels_pkl_path=None, vrep_geom=None, save_full_3d=False,
                 expand_octree_points=False, contact_pkl_path=None, 
                 convert_to_dense_voxels=False, min_xyz=MIN_XYZ, max_xyz=MAX_XYZ):
        self.convert_to_dense_voxels = convert_to_dense_voxels
        super(OctreePickleLoader, self).__init__(voxels_list=None, 
                                                 min_xyz=min_xyz,
                                                 max_xyz=max_xyz)
        self.voxels_pkl_path = voxels_pkl_path
        self.vrep_geom = vrep_geom
        self.contact_pkl_path = contact_pkl_path

        self.anchor_index_int = None
        self.other_index_int = None

        self.expand_octree_points = expand_octree_points
        self.other_obj_voxels_idx = None

        self.save_full_3d = save_full_3d

        if contact_pkl_path is not None:
            self.contact_data = self.read_contact_data(contact_pkl_path)

    def read_contact_data(self, contact_pkl_path):
        assert os.path.exists(contact_pkl_path), 'Contact pkl does not exist'
        with open(contact_pkl_path, 'rb') as contact_f:
            contact_data = pickle.load(contact_f)
        return contact_data
    
    def get_contact_points_as_array(self, before=True):
        key = 'before_contact' if before else 'after_contact'
        if self.contact_data[key] is None:
            return None
        contact_list = self.contact_data[key]
        contact_data = [c[2:5] for c in contact_list]
        return np.array(contact_data)
    
    def get_contact_points(self, before=True):
        key = 'before_contact' if before else 'after_contact'
        if self.contact_data[key] is None:
            return {}
        contact_list = self.contact_data[key]
        contact_info = {'num_contacts': len(contact_list), 'contacts': []}
        for i, c_data in enumerate(contact_list):
            contact_info['contacts'].append({
                'point': c_data[2:2+3],
                'force': c_data[5:5+3],
                'normal': c_data[8:8+3],
            })
        return contact_info

    def init_voxel_index(self, voxels_key='voxels_before', before=True):
        with open(self.voxels_pkl_path, 'rb') as pkl_f:
            data = pickle.load(pkl_f)
        other_voxels = np.array(data[voxels_key]['other']).reshape(-1, 3)
        anchor_voxels = np.array(data['voxels_before']['anchor']).reshape(-1, 3)
    
        # Get dense anchor voxels??
        if not self.convert_voxel_index_to_3d_index:
            raise ValueError("Already dense")
            anchor_voxels = self.vrep_geom.create_dense_voxels_from_sparse(
                anchor_voxels, before=True, anchor_obj=True)
            other_voxels = self.vrep_geom.create_dense_voxels_from_sparse(
                other_voxels, before=True, anchor_obj=False)

        # import ipdb; ipdb.set_trace()
        anchor_idx = (anchor_voxels - self.min_xyz) / self.voxel_size
        self.anchor_index_int = np.around(anchor_idx, decimals=1).astype(np.int32)
        other_idx = (other_voxels - self.min_xyz) / self.voxel_size
        self.other_index_int = np.around(other_idx, decimals=1).astype(np.int32)

        if np.max(self.other_index_int[:, 0]) >= self.xyz_size[0] or \
           np.max(self.other_index_int[:, 1]) >= self.xyz_size[1] or \
           np.max(self.other_index_int[:, 2]) >= self.xyz_size[2]:
            print(f"Anchor and other are too far: {self.voxels_pkl_path}")
            return False

        if self.expand_octree_points and self.vrep_geom is not None:
            bb = self.vrep_geom.get_bounding_box_in_object_frame(
               before=before, anchor_obj=False)
            p = self.vrep_geom.get_position_for_object(
                before=before, anchor_obj=False)
            bb_low = np.floor(bb[0] / self.voxel_size).astype(np.int32)
            bb_high = np.floor(bb[1] / self.voxel_size).astype(np.int32)
            idx = np.meshgrid(np.arange(bb_low[0], bb_high[0]),
                              np.arange(bb_low[1], bb_high[1]),
                              np.arange(bb_low[2], bb_high[2]))
            idx = np.stack(idx).reshape(3, -1)
            # idx = np.vstack([idx, np.linalg.norm(idx, axis=0)])
            idx = np.vstack([idx, np.ones(idx.shape[1])])
            T_other = self.vrep_geom.get_transformation_for_object(
                    before=before, anchor_obj=False)
            # Make positions in transformation 0
            T_other[:3, -1] = (T_other[:3, -1] - self.min_xyz) / self.voxel_size
            idx_transf = np.around(np.dot(T_other, idx)).astype(np.int32)
            self.other_obj_voxels_idx = np.array(idx_transf[:3, :].T)

        if self.save_full_3d:
            self.status_3d, self.full_3d = self.parse()     
            # Remove the other data to make up space
            self.anchor_index_int = None
            self.other_index_int = None
            self.other_obj_voxels_idx = None

        return True    

    def parse(self):
        '''Valid octree that fits into the above memory.'''
        if self.save_full_3d and self.full_3d is not None:
            return self.status_3d, self.full_3d

        if self.full_3d is None:
            # start = time.time()
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

            if self.expand_octree_points and self.other_obj_voxels_idx is not None:
                min_anchor_idx = self.anchor_index_int.min(axis=0)
                max_anchor_idx = self.anchor_index_int.max(axis=0)
                full_3d[0, 
                        min_anchor_idx[0]+1:max_anchor_idx[0]-1,
                        min_anchor_idx[1]+1:max_anchor_idx[1]-1,
                        min_anchor_idx[2]+1:max_anchor_idx[2]-1] = 1
                full_3d[1, 
                        min_anchor_idx[0]+1:max_anchor_idx[0]-1,
                        min_anchor_idx[1]+1:max_anchor_idx[1]-1,
                        min_anchor_idx[2]+1:max_anchor_idx[2]-1] = 1

                full_3d[0, 
                        self.other_obj_voxels_idx[:, 0],
                        self.other_obj_voxels_idx[:, 1],
                        self.other_obj_voxels_idx[:, 2]] = 1
                full_3d[2, 
                        self.other_obj_voxels_idx[:, 0],
                        self.other_obj_voxels_idx[:, 1],
                        self.other_obj_voxels_idx[:, 2]] = 1

            # end = time.time()
        
        return True, full_3d

class CanonicalOctreeVoxels(object):
    def __init__(self, voxel_pkl_path, add_size_channels=False):
        self.voxel_pkl_path = voxel_pkl_path
        self.voxel_size = 0.025
        self.max_voxel_size = (32, 32, 16)
        self.voxel_data = None
        self.final_voxel_data = None
        self.add_size_channels = add_size_channels
        self.zoom_mult = None

    def init_voxel_index(self, voxels_key='voxels_before'): 
        with open(self.voxel_pkl_path, 'rb') as pkl_f: 
            data = pickle.load(pkl_f)
        other_voxels = np.array(data[voxels_key]['other']).reshape(-1, 3)
        anchor_voxels = np.array(data['voxels_before']['anchor']).reshape(-1, 3)

        min_xyz_anchor = np.min(anchor_voxels, axis=0)
        min_xyz_other = np.min(other_voxels, axis=0)

        max_xyz_anchor = np.max(anchor_voxels, axis=0)
        max_xyz_other = np.max(other_voxels, axis=0)

        min_xyz =  np.minimum(min_xyz_anchor, min_xyz_other)
        max_xyz = np.maximum(max_xyz_anchor, max_xyz_other)

        xyz_size = np.around((max_xyz - min_xyz) / self.voxel_size, 
                             decimals=1)
        # max_xyz is not included in xyz_size hence add 1
        xyz_size = xyz_size.astype(np.int32) + 1

        if self.add_size_channels:
            voxel_data = np.zeros([6] + xyz_size.tolist())
        else:
            voxel_data = np.zeros([3] + xyz_size.tolist())
        anchor_idx = self.convert_voxel_index_to_3d_index(
            anchor_voxels, min_xyz)
        other_idx = self.convert_voxel_index_to_3d_index(
            other_voxels, min_xyz)
        
        voxel_data[0][anchor_idx[:, 0], 
                      anchor_idx[:, 1],
                      anchor_idx[:, 2]] = 1
        voxel_data[0][other_idx[:, 0], 
                      other_idx[:, 1],
                      other_idx[:, 2]] = 1
        voxel_data[1][anchor_idx[:, 0], 
                      anchor_idx[:, 1],
                      anchor_idx[:, 2]] = 1
        voxel_data[2][other_idx[:, 0], 
                      other_idx[:, 1],
                      other_idx[:, 2]] = 1
        if self.add_size_channels:
            voxel_data[3][other_idx[:, 0],
                          other_idx[:, 1],
                          other_idx[:, 2]] = 0.025*other_idx[:, 0]
            voxel_data[4][other_idx[:, 0],
                          other_idx[:, 1],
                          other_idx[:, 2]] = 0.025*other_idx[:, 1]
            voxel_data[5][other_idx[:, 0],
                          other_idx[:, 1],
                          other_idx[:, 2]] = 0.025*other_idx[:, 2]
        
        self.voxel_data = voxel_data
        self.final_voxel_data, self.zoom_mult = self.reshape_voxel_data(
            voxel_data, self.max_voxel_size)
        self.min_xyz = min_xyz
        self.max_xyz = max_xyz
        return True

    def convert_voxel_index_to_3d_index(self, xyz_arr, min_xyz):
        idx = (xyz_arr - min_xyz) / self.voxel_size
        idx_int = np.around(idx, decimals=1).astype(np.int32)
        return idx_int
    
    def reshape_voxel_data(self, voxel_data, desired_shape):
        desired_data = []
        zoom_mult_final = None
        for c in range(voxel_data.shape[0]):
            d, zoom_mult = image_utils.zoom_array(voxel_data[c], desired_shape)
            desired_data.append(d)
            if zoom_mult_final is None:
                zoom_mult_final = zoom_mult
            else:
                for i in range(len(zoom_mult)):
                    if abs(zoom_mult_final[i] - zoom_mult[i]) > 1e-4:
                        print(zoom_mult)
                        print(zoom_mult_final)
                        raise ValueError("zoom multipler should be same for all channels.")

        # import ipdb; ipdb.set_trace()
        return np.stack(desired_data), zoom_mult_final

    def parse(self):
        # return True, self.voxel_data
        return True, self.final_voxel_data
