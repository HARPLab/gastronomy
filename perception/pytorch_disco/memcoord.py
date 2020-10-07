import torch
# import utils
import numpy as np
# import utils_vox

class Coord:
    def __init__(self, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, FLOOR=0, CEIL=-5):
        self.XMIN = XMIN # right (neg is left)
        self.XMAX = XMAX # right
        self.YMIN = YMIN # down (neg is up)
        self.YMAX = YMAX # down
        self.ZMIN = ZMIN
        self.ZMAX = ZMAX
        self.FLOOR = FLOOR # objects don't dip below this
        self.CEIL = CEIL # objects don't rise above his
        self.values = torch.from_numpy(np.asarray([self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX, self.FLOOR, self.CEIL]))

    def __repr__(self):
        return str([self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX, self.FLOOR, self.CEIL])


class VoxCoord:
    """
    To define a memory, you need to define a coordinate system
    and the size of the memory (proto)
    """
    def __init__(self, coord, proto, correction=True):
        """
        Don't touch the correction. This is for testing file using the discovery repo as ground truth
        """
        self.proto = proto
        self.coord = coord
        self.correction=correction
        self.vox_T_cam = None
        self.cam_T_vox = None

        self.build(self.coord, self.proto)

    def __repr__(self):
        return f'proto: {self.proto} \ncoord: {self.coord}'

    def build(self, coord, proto):
        if self.vox_T_cam == None:
            X, Y, Z = self.proto.shape[1], self.proto.shape[0], self.proto.shape[2]
            self.vox_T_cam = utils_vox.get_mem_T_ref(1, Z, Y, X)
        if self.cam_T_vox == None:
            self.cam_T_vox = torch.inverse(self.vox_T_cam)
