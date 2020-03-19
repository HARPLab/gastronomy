import numpy as np


def get_xy_unit_vector_for_angle(theta):
    return [np.cos(theta), np.sin(theta), 0]


def are_position_similar(p, q, error_threshold=0.004):
    assert len(p) == 3 and len(q) == 3, "Invalid positions"
    dist = np.array([abs(p[i] - q[i]) for i in range(3)])
    return np.all(dist < error_threshold)

def sample_exactly_around_edge(start, end, half_size, region=None):
    '''Sample value around the edges.

    Should intersect with (start, end), but should also be outside. 
    '''
    assert region is None or region in ['low', 'high']
    sample_region = region
    if region is not None:
        if 'low' in region:
            region_idx = 0
        elif 'high' in region:
            region_idx = 1
        else:
            raise ValueError("Invalid region value: {}".format(region))
    else:
        region_idx = np.random.randint(0, 2) % 2
    
    npu = np.random.uniform
    if region_idx == 0:
        # pos = npu(start - 2*half_size + 0.01, start-0.001) + half_size
        # This line below will have pos close to the edge which is stable, what
        # we want is far
        # pos = npu(start-half_size/3.0, start+half_size/3.0)
        
        # this line should give us far
        pos = npu(start-half_size+half_size/5.0, start+half_size/5.0)
        sample_region = 'low'
    elif region_idx == 1:
        # pos = npu(end+0.001, end + 2*half_size - 0.01) - half_size
        # This line below will have pos close to the edge which is stable, what
        # we want is far
        # pos = npu(end-half_size/3.0, end+half_size/3.0)

        # this line should give us far
        pos = npu(end-half_size/5.0, end+half_size-half_size/5.0)
        sample_region = 'high'
    else:
        raise ValueError("Invalid region idx when sampling pos around edge")

    return pos, {'sample_region': sample_region}


def sample_exactly_outside_edge(start, end, threshold, half_size, region=None,
                                min_threshold=0.001):
    '''Sample value outside the edges.

    Should not intersect with (start, end), but should be just outside.
    '''
    region_idx = np.random.randint(0, 2) % 2
    assert region is None or region in ['low', 'high']
    sample_region = region
    if region is not None:
        if 'low' in region:
            region_idx = 0
        elif 'high' in region:
            region_idx = 1
        else:
            raise ValueError("Invalid region value: {}".format(region))
    else:
        region_idx = np.random.randint(0, 2) % 2

    npu = np.random.uniform
    dist = npu(min_threshold, threshold)
    if region_idx == 0:
        pos = start - dist - half_size
        sample_region = 'low'
    elif region_idx == 1:
        pos = end + dist + half_size
        sample_region = 'high'
    else:
        raise ValueError("Invalid region idx when sampling pos around edge")

    return pos, {'sample_region': sample_region}


def sample_from_edges(start, end, threshold, region=None):
    '''Sample value from the edge of start and end.

    start: Float.
    end: Float.
    region: str. Value of region to sample from. Value values 
        ['inside', 'outside']
    '''
    th = threshold
    assert region is None or region in ['inside', 'outside', 'inside_low', 
                                        'inside_high', 'outside_low', 
                                        'outside_high']
    if region is not None and 'inside' in region:
        region_1 = (start, start+th)
        region_2 = (end-th, end)
    elif region is not None and 'outside' in region:
        region_1 = (start-th, start)
        region_2 = (end, end+th)
    else:
        region_1 = (start-th, start+th)
        region_2 = (end-th, end+th)
    
    npu = np.random.uniform
    npri = np.random.randint

    info = {}
    if region is not None and 'low' in region:
        sample_reg = region_1
        info['region'] = 'low'
    elif region is not None and 'high' in region:
        sample_reg = region_2
        info['region'] = 'high'
    else:
        if npri(2) % 2 == 0:
            sample_reg = region_1
            info['region'] = 'low'
        else:
            sample_reg = region_2
            info['region'] = 'high'
    
    return npu(sample_reg[0], sample_reg[1]), info


def create_dense_voxels_from_sparse(obj_voxel_points, obj_transform):
    # Take inverse of transform to convert everything in the object space. 
    # This should be axis aligned.
    T_inv = np.linalg.inv(obj_transform)
    obj_voxel_arr = np.array(obj_voxel_points).reshape(-1, 3)
    sparse_voxels = np.hstack([obj_voxel_arr, 
                               np.ones((obj_voxel_arr.shape[0], 1))])
    transf_sparse_voxels = np.dot(T_inv, sparse_voxels.T)
    transf_sparse_voxels = transf_sparse_voxels.T
    [min_x, min_y, min_z, _] = np.min(transf_sparse_voxels, axis=0)
    [max_x, max_y, max_z, _] = np.max(transf_sparse_voxels, axis=0)
    curr_z, step_z = min_z, 0.005
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

    dense_obj_voxel_points = org_dense_obj_voxels_arr.reshape(-1).tolist()
    return dense_obj_voxel_points 


def get_transformation_matrix_for_vrep_transform_list(transform):
    T = transform + [0, 0, 0, 1]
    return np.array(T).reshape(4, 4)


class Sat3D(object):
    def __init__(self, lb_a, ub_a, lb_b, ub_b, Ta, Tb):
        self.lb_a = np.array(lb_a)
        self.ub_a = np.array(ub_a)
        self.lb_b = np.array(lb_b)
        self.ub_b = np.array(ub_b)
        self.Ta = Ta
        self.Tb = Tb
        self.Ta_m = np.array(Ta).reshape(-1, 4)
        self.Tb_m = np.array(Tb).reshape(-1, 4)

    def get_cuboid_axes(self, lb, ub):
        e11, e12 = [ub[0] - lb[0], 0, 0], [0, ub[1]-lb[1], 0]
        ax1 = np.cross(e11, e12)

        e21, e22 = [ub[0] - lb[0], 0, 0], [0, 0, ub[2]-lb[2]]
        ax2 = np.cross(e21, e22)

        e31, e32 = [0, ub[1]-lb[1], 0], [0, 0, ub[2]-lb[2]]
        ax3 = np.cross(e31, e32)

        return ax1, ax2, ax3

    def get_transform_cuboid_axes(self, lb, ub, T=None):
        e1 = [ub[0]-lb[0], 0, 0]
        e2 = [0, ub[1]-lb[1], 0]
        e3 = [0, 0, ub[2]-lb[2]]
        if T is not None:
            return self.transform_vec(T, *[e1, e2, e3])
        else:
            return [e1, e2, e3]
    
    def transform_points(self, T, *args):
        t_args = []
        for v in args:
            v_4 = None
            if len(v) == 3:
                if type(v) is np.ndarray:
                    v_4 = v.tolist() + [1]
                else:
                    v_4 = list(v) + [1]
            else:
                assert len(v_4) == 4, "Incorrect vec size not 4."
                v_4 = v
            new_v = np.dot(T, np.array(v_4).reshape(-1, 1))
            # assert abs(new_v[-1, 0]-1.0) <= 1e-4
            t_args.append(new_v[:, 0])
        return t_args

    def transform_vec(self, T, *args):
        t_args = []
        for v in args:
            assert len(v) == 3
            if T.shape[1] == 4:
                new_v = np.dot(T[:3, :3], np.array(v).reshape(-1, 1))
            else:
                new_v = np.dot(T, np.array(v).reshape(-1, 1))
            t_args.append(new_v[:, 0])
        return t_args

    def get_inter_cuboid_axes(self):
        ax_a_list = self.get_transform_cuboid_axes(
            self.lb_a, self.lb_b, T=self.Ta_m)
        ax_b_list = self.get_transform_cuboid_axes(
            self.lb_a, self.lb_b, T=self.Tb_m)

        ax_list = []
        for ax_a in ax_a_list:
            for ax_b in ax_b_list:
                ax_list.append(np.cross(ax_a, ax_b))
        return ax_list

    def get_all_cuboid_vertices(self, lb, ub, T=None):
        vertices = [
            (lb[0], lb[1], lb[2]),
            (ub[0], ub[1], lb[2]),
            (lb[0], ub[1], lb[2]),
            (lb[1], ub[0], lb[2]),
            (lb[0], lb[1], ub[2]),
            (ub[0], ub[1], ub[2]),
            (lb[0], ub[1], ub[2]),
            (lb[1], ub[0], ub[2]),
        ]
        if T is not None:
            return self.transform_points(T, *vertices)
        else:
            return vertices

    def normalize_vec(self, v):
        if np.linalg.norm(v) == 0:
            return v
        return v / np.linalg.norm(v)

    def get_all_axes_distance(self):
        ax_list_a = self.get_cuboid_axes(self.lb_a, self.ub_a)
        ax_list_a = [self.normalize_vec(ax) for ax in ax_list_a]
        ax_list_a_transf = self.transform_vec(self.Ta_m, *ax_list_a)

        ax_list_b = self.get_cuboid_axes(self.lb_b, self.ub_b)
        ax_list_b = [self.normalize_vec(ax) for ax in ax_list_b]
        ax_list_b_transf = self.transform_vec(self.Tb_m, *ax_list_b)

        cross_ax_list = self.get_inter_cuboid_axes()

        all_ax = ax_list_a_transf + ax_list_b_transf + cross_ax_list
        all_ax_norm = [self.normalize_vec(ax) for ax in all_ax]

        vert_a = self.get_all_cuboid_vertices(self.lb_a, self.ub_a, self.Ta_m)
        vert_b = self.get_all_cuboid_vertices(self.lb_b, self.ub_b, self.Tb_m)

        ax_dist = []
        for ax in all_ax_norm:
            vert_a_dist = [np.dot(v, ax) for v in vert_a]
            vert_b_dist = [np.dot(v, ax) for v in vert_b]
            min_a, max_a = min(vert_a_dist), max(vert_a_dist)
            min_b, max_b = min(vert_b_dist), max(vert_b_dist)

            if max_a <= min_b:
                dist = min_b - max_a
            elif max_b <= min_a:
                dist = min_a - max_b
            else:
                # Overlap
                dist = -1
            ax_dist.append(dist)

        return ax_dist

if __name__ == '__main__':

    # a = [(-1, -1), (1, 1)]
    # T_a = [1, 0, 0, 0, 1, 0]
    # b = [(-2, -2), (2, 2)]
    # T_b [1, 1, 3, -1, 1, 3]
    sat = Sat3D(
        [-1, -1, -1], [1, 1, 1], [-1, -1, -1], [1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 6, -1, 1, 0, 6, 0, 0, 1, 0]
    )
    d = sat.get_all_axes_distance()
    print("dist: {}".format(d))

