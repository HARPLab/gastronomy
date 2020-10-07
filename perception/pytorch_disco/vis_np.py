import binvox_rw as binvox_rw
import numpy as np


def binarize(a, thres=0.5):
  a[a>thres] = 1
  a[a<thres] = 0
  return a 

def save_voxel(voxel_, filename, THRESHOLD=0.5):
  S1 = voxel_.shape[2]
  S2 = voxel_.shape[1]
  S3 = voxel_.shape[0]

  binvox_obj = binvox_rw.Voxels(
    np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
    dims = [S1, S2, S3],
    translate = [0.0, 0.0, 0.0],
    scale = 1.0,
    axis_order = 'xyz'
  )   

  with open(filename, "wb") as f:
    binvox_obj.write(f)


def plot_cam(ax_, cam_matrix, color="r", msg=None, length=0.05):
    # see here for example usage
    # https://github.com/ricsonc/3dmapping/blob/htung-view-predict/setup/parse_real.py#L52
    camera_xyz = cam_matrix[:3, 3]
    x_dir = cam_matrix[:3, 0]
    y_dir = cam_matrix[:3, 1]
    z_dir = cam_matrix[:3, 2]    
    ax_.scatter(camera_xyz[0], camera_xyz[1] , camera_xyz[2], c=color, marker='o')
    ax_.text(camera_xyz[0], camera_xyz[1], camera_xyz[2], msg)

    ax_.quiver(camera_xyz[0], camera_xyz[1], camera_xyz[2], x_dir[0], x_dir[1], x_dir[2], color="r", length=length)
    ax_.quiver(camera_xyz[0], camera_xyz[1], camera_xyz[2], y_dir[0], y_dir[1], y_dir[2], color="g", length=length)
    ax_.quiver(camera_xyz[0], camera_xyz[1], camera_xyz[2], z_dir[0], z_dir[1], z_dir[2], color="b", length=length)