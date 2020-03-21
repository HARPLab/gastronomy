import numpy as np
import torch
# from lib_classes import Nel_Utils as nlu
# import hyperparams as hyp
import ipdb
st = ipdb.set_trace
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
'''
http://www.cs.cmu.edu/~16385/s17/Slides/5.4_Generalized_Hough_Transform.pdf
http://www.cs.toronto.edu/~fidler/slides/2015/CSC420/lecture17.pdf
https://pdfs.semanticscholar.org/5b57/ee9492713c9249da291deb1b3d3ca91febbb.pdf

'''

class HoughTransform(object):
    def __init__(self):
        # 
        self.threshold = 0.5
        self.num_sample_points = 20
        self.EPS = 1e-4
        self.r_tables = list()
        self.vis=False
        self.debug=False
        # self.offsets = [2, 1, 0, -1, -2]
        self.offsets = [0]

    def get_closest_points(self, z, y, x, surface_indices, num_pts=5):
        zpts = surface_indices[0]
        ypts = surface_indices[1]
        xpts = surface_indices[2]
        zpts = (zpts - z)**2
        ypts = (ypts - y)**2
        xpts = (xpts - x)**2
        dist = zpts + ypts + xpts
        closest_pts = np.argsort(dist)
        closest_pts =  closest_pts[:num_pts]
        np.random.shuffle(closest_pts) # This can probably help prevent collinear points early.
        return closest_pts

    '''
    https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d
    '''
    def get_plane(self, z, y, x, surface_indices):
        while True:
            closest_points = self.get_closest_points(z, y, x, surface_indices)
            # p0, p1, p2 = closest_points[0], closest_points[1], closest_points[2]
            i0, i1, i2 = closest_points[0], closest_points[1], closest_points[2]
            p0 = (surface_indices[0][i0], surface_indices[1][i0], surface_indices[2][i0])
            p1 = (surface_indices[0][i1], surface_indices[1][i1], surface_indices[2][i1])
            p2 = (surface_indices[0][i2], surface_indices[1][i2], surface_indices[2][i2])

            z0, y0, x0 = p0
            z1, y1, x1 = p1
            z2, y2, x2 = p2

            # print("Points selected: ", p0, p1, p2)

            ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
            vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

            point  = np.array([x0, y0, z0])
            normal = np.array(u_cross_v)
            invalid_plane = True
            # st()
            '''
            Validate that all 3 points are not collinear.
            '''
            for i in normal:
                if i != 0:
                    invalid_plane = False
            if not invalid_plane:
                break
            
            # st()

        d = -point.dot(normal)
        # print("Normal : ", normal)
        # print("D: ", d)

        # st()
        if self.vis:
            xx, yy = np.meshgrid(range(10), range(10))
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            plt3d = plt.figure().gca(projection='3d')
            plt3d.plot_surface(xx, yy, z)
            plt.show(block=True)
        
        normal = np.array([normal[2], normal[1], normal[0]]) # get normal back in zyx format
        normal = normal/np.linalg.norm(normal) #normalize the normal for angle calculations.
        return normal

    def create_single_object_r_table(self, voxel):
        voxel_r_table = dict()
        z_vector = np.array([1, 0, 0])
        x_vector = np.array([0, 0, 1])
        surface_indices = np.where(voxel > self.threshold)
        internal_pt = (np.average(surface_indices[0])+self.EPS, np.average(surface_indices[1])+self.EPS, np.average(surface_indices[2])+self.EPS)
        for z, y, x in zip(surface_indices[0], surface_indices[1], surface_indices[2]):
            plane = self.get_plane(z,y,x,surface_indices)
            z_angle = np.degrees(np.arccos(np.dot(plane, z_vector)))
            x_angle = np.degrees(np.arccos(np.dot(plane, x_vector)))
            r = np.linalg.norm(np.asarray([z,y,x]) - internal_pt)
            
            alpha_radian = np.arccos((internal_pt[0] - z)/r)
            alpha = np.degrees(alpha_radian)
            beta_radian = (np.arccos((internal_pt[2] - x)/(r*np.sin(alpha_radian))))
            beta = np.degrees(beta_radian)
            # print("R: ", r)
            # print("z_angle: ", z_angle)
            # print("x_angle: ", x_angle)
            # print("alpha: ", alpha)
            # print("beta: ", beta)
            # print("internal point: ", internal_pt)
            # print("current point: ", (z, y, x))
            # st()
            x_angle, z_angle = np.int(x_angle), np.int(z_angle)
            if (x_angle, z_angle) not in voxel_r_table:
                voxel_r_table[(x_angle, z_angle)] = list()
            voxel_r_table[(x_angle, z_angle)].append((internal_pt, r, alpha_radian, beta_radian))


        return voxel_r_table

    def instantiate_hough_transform(self, voxels):
        '''
        torch.Size([4, 1, 72, 72, 72])
        N will be number of distinct object templates we have
        '''
        N, C, D, H, W = voxels.shape
        # st()
        for i in range(N):
            voxel_r_table = self.create_single_object_r_table(voxels[i, 0])
            self.r_tables.append(voxel_r_table)
        # st()

    def query_r_tables_for_given_voxel(self, voxel):
        
        z_vector = np.array([1, 0, 0])
        x_vector = np.array([0, 0, 1])
        surface_indices = np.where(voxel > self.threshold)
        selected_indices = np.random.permutation(np.arange(len(surface_indices[0])))[: self.num_sample_points]
        '''
        We'll be querying on some indices to save computation costs.
        '''
        surface_indices_sampled = (surface_indices[0][selected_indices], surface_indices[1][selected_indices], surface_indices[2][selected_indices])
        for r_table in self.r_tables:
            aux_table = dict()
            max_count = 0
            max_center = (0,0,0)
            for z, y, x in zip(surface_indices_sampled[0], surface_indices_sampled[1], surface_indices_sampled[2]):
                plane = self.get_plane(z,y,x,surface_indices)
                z_angle = np.degrees(np.arccos(np.dot(plane, z_vector)))
                x_angle = np.degrees(np.arccos(np.dot(plane, x_vector)))
                
                if (x_angle, z_angle) in r_table:
                    for r_alpha_beta in r_table[(x_angle, z_angle)]:
                        _, r, alpha, beta = r_alpha_beta
                        # st()
                        # print("r, alpha, beta: ", r, alpha, beta)
                        zc = int(z + r*np.cos(alpha))
                        yc = int(y + r*np.sin(alpha)*np.sin(beta))
                        xc = int(x + r*np.sin(alpha)*np.cos(beta))
                        for zoff in self.offsets:
                            for yoff in self.offsets:
                                for xoff in self.offsets:
                                    zo = zc + zoff
                                    yo = yc + yoff
                                    xo = xc + xoff
                                    print("point considered internal: ", zo, yo, xo)
                                    if (xo, yo, zo) in aux_table:
                                        aux_table[(xo, yo, zo)] += 1
                                    else:
                                        aux_table[(xo, yo, zo)] = 1
                                    if aux_table[(xo, yo, zo)] > max_count:
                                        max_count = aux_table[(xo, yo, zo)]
                                        max_center = (xo, yo, zo)
            print("max count: ", max_count)

    
    def query_r_tables(self, voxels):
        '''
        torch.Size([B, 1, 72, 72, 72])
        '''
        B, C, D, H, W = voxels.shape
        for i in range(B):
            aux_table = self.query_r_tables_for_given_voxel(voxels[i, 0])


def get_voxel_cube(vis=False):
    
    N1 = 40
    N2 = 40
    N3 = 40
    # ma = np.random.choice([0,1], size=(N1,N2,N3), p=[0.99, 0.01])
    ma = np.zeros((N1, N2, N3))
    ma[3:8, 3:8, 3:8] = 1
    ma[4:7, 4:7, 4:7] = 0

    ma = ma.transpose(2,1,0)
    if vis:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_aspect('equal')
        plt.xlabel('xlabel', fontsize=18)
        plt.ylabel('ylabel', fontsize=16)
        ax.voxels(ma, edgecolor="k")
        plt.show(block=True)
    
    na = np.zeros((N1, N2, N3))
    center = [20, 20, 20]
    radius = 10
    for x in range(N1):
        for y in range(N2):
            for z in range(N3):
                pt = np.array([x, y, z])
                if np.linalg.norm([pt-center]) >= radius-1 and np.linalg.norm([pt-center])<=radius+1:
                # if np.linalg.norm([pt-center]) == radius:
                    na[x, y, z] = 1

    na = na.transpose(2,1,0)
    if True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_aspect('equal')
        plt.xlabel('xlabel', fontsize=18)
        plt.ylabel('ylabel', fontsize=16)
        ax.voxels(na, edgecolor="k")
        plt.show(block=True)
    vox = np.stack([ma, na], axis=0)
    vox = np.expand_dims(vox, axis=1)
    
    # return ma.transpose(2,1,0)
    return vox

if __name__ == '__main__':
    ht = HoughTransform()
    vox = get_voxel_cube()
    # st()
    ht.instantiate_hough_transform(vox)
    ht.query_r_tables(vox)
    # st()





        
