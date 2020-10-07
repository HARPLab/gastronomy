
import argparse
import numpy as np
import os
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "replica_multiview_builder"
os.environ["run_name"] = "check"
import ipdb 
st = ipdb.set_trace
import open3d as o3d
import utils_geom 
import lib_classes.Nel_Utils as nlu 
import torch
from colmap_read_write_model import read_model

# python read_depth_camera.py -d /Users/shamitlal/Desktop/temp/colmap_outs/desk_imgs/dense/0/stereo/depth_maps/IMG_0686.jpg.photometric.bin 
# python setup/colmap_parse.py -d /Users/shamitlal/Desktop/temp/colmap_outs/desk_imgs/dense/0/stereo/depth_maps/IMG_0686.jpg.photometric.bin input_model /Users/shamitlal/Desktop/temp/colmap_outs/desk_imgs/sparse/0 input_format .bin

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--depth_map",
                        help="path to depth map", type=str, required=True)
    parser.add_argument("--min_depth_percentile",
                        help="minimum visualization depth percentile",
                        type=float, default=5)
    parser.add_argument("--max_depth_percentile",
                        help="maximum visualization depth percentile",
                        type=float, default=95)
    parser.add_argument('--input_model', help='path to input model folder')
    parser.add_argument('--input_format', choices=['.bin', '.txt'],
                        help='input model format')
    args = parser.parse_args()
    return args

def get_intrinsics(cameras, curx, cury):
    p = cameras[1].params
    height, width = cameras[1].height, cameras[1].width
    batched_list = []
    for i in range(len(p)):
        batched_param = torch.tensor(p[i]).unsqueeze(0).unsqueeze(0)
        batched_list.append(batched_param)
    fx, fy, cx, cy = batched_list[0], batched_list[1], batched_list[2], batched_list[3]
    # st()
    pix_T_camX = utils_geom.pack_intrinsics(fx, fy, cx, cy)
    pix_T_camX = utils_geom.scale_intrinsics(pix_T_camX, (1.*curx)/width, (1.*cury)/height)
    return pix_T_camX

def main():
    args = parse_args()

    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError("min_depth_percentile should be less than or equal "
                         "to the max_depth_perceintile.")

    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(args.depth_map):
        raise FileNotFoundError("File not found: {}".format(args.depth_map))

    cameras, images, points3D = read_model(path=args.input_model, ext=args.input_format)


    depth_map = read_array(args.depth_map)
    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth_percentile, args.max_depth_percentile])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    pix_T_camX = get_intrinsics(cameras, depth_map.shape[1], depth_map.shape[0])

    # Visualize the depth map.
    import pylab as plt
    plt.figure()
    plt.imshow(depth_map)
    plt.title('depth map')
    plt.show(block=True)
    # st()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    xyz_camXs = utils_geom.depth2pointcloud_cpu(torch.tensor(depth_map).unsqueeze(0).unsqueeze(0), torch.tensor(pix_T_camX)).squeeze(0).numpy()
    # xyz_camXs = xyz_camXs[np.where(xyz_camXs[:,-1]<6)]
    # st()
    pcd_camX = nlu.make_pcd(xyz_camXs)
    o3d.visualization.draw_geometries([pcd_camX, mesh_frame])
    
    


if __name__ == "__main__":
    main()