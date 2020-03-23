import numpy as np
import argparse
import os

from utilities import vrep_utils as vu

from lib import vrep


def ms_add_3d_lists(v1, v2):
    assert len(v1) == 3 and len(v2) == 3
    return [v1[i] + v2[i] for i in range(3)]

npu = np.random.uniform
npri = np.random.randint

class VrepScene(object):
    def __init__(self, args):
        self.args = args
        # just in case, close all opened connections
        vrep.simxFinish(-1)
        self.client_id = self.connect(args.scene_file)
        self.handles_dict = self.get_handles()

    def stop(self):
        vu.stop_sim(self.client_id)

    def stop_and_finish(self):
        self.stop()
        vrep.simxFinish(self.client_id)

    def reset(self):
        self.stop()
        self.start()

    def connect(self, scene_file):
        client_id = vu.connect_to_vrep()
        res = vu.load_scene(client_id, scene_file)
        return client_id

    def start(self):
        vu.start_sim(self.client_id)

    def step(self):
        vu.step_sim(self.client_id)

    def get_handles(self):
        cam_rgb_names = ["Camera"]
        cam_rgb_handles = [vrep.simxGetObjectHandle(
            self.client_id, n, vrep.simx_opmode_blocking)[1] for n in cam_rgb_names]

        octree_handle = vu.get_handle_by_name(self.client_id, 'Octree')
        anchor_obj_handle = vu.get_handle_by_name(self.client_id, 'Cuboid0')
        other_obj_handle = vu.get_handle_by_name(self.client_id, 'Cuboid')

        return dict(
            anchor_obj=anchor_obj_handle,
            other_obj=other_obj_handle,
            octree=octree_handle,
            cam_rgb=cam_rgb_handles,
        )

    def get_absolute_bb(self, obj_handle):
        anchor_bb = self.get_object_bb(obj_handle)
        anchor_pos = self.get_object_position(obj_handle)
        lb = anchor_bb[0] + anchor_pos
        ub = anchor_bb[1] + anchor_pos
        return lb, ub

    def get_object_bb(self, obj_handle):
        value_list = []
        for param in [15, 16, 17, 18, 19, 20]:
            resp, value = vrep.simxGetObjectFloatParameter(
                self.client_id, obj_handle, param,
                vrep.simx_opmode_blocking)
            value_list.append(value)

        return np.array(value_list[0:3]), np.array(value_list[3:])

    def set_object_position(self, obj_handle, position):
        resp = vrep.simxSetObjectPosition(
            self.client_id,
            obj_handle,
            -1,  # Sets absolute position
            position,
            vrep.simx_opmode_oneshot)
        if resp != vrep.simx_return_ok:
            print("Error in setting object position: {}".format(resp))

    def set_object_orientation(self, obj_handle, euler_angles):
        resp = vrep.simxSetObjectOrientation(
            self.client_id,
            obj_handle,
            -1,  # Sets absolute orientation
            euler_angles,
            vrep.simx_opmode_oneshot)
        if resp != vrep.simx_return_ok:
            print("Error in setting object orientation: {}".format(resp))

    def get_object_position(self, handle, relative_to_handle=-1):
        '''Return the object position in reference to the relative handle.'''
        response, position = vrep.simxGetObjectPosition(
            self.client_id, handle, relative_to_handle, 
            vrep.simx_opmode_blocking)
        if response != vrep.simx_return_ok:
            print("Error: Cannot query position for handle {} "
                  "with reference to {}".format(handle, relative_to_handle))
        return np.array(position)

    def get_object_orientation(self, handle, relative_to_handle=-1):
        '''Return the object orientation in reference to the relative handle.'''
        response, orientation = vrep.simxGetObjectOrientation(
            self.client_id, handle, relative_to_handle,
            vrep.simx_opmode_blocking)
        if response != 0:
            print("Error: Cannot query position for handle {} with reference to {}".
                  format(handle, reference_handle))
        return np.array(orientation)

    def get_octree_voxels(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_GetOctreeVoxels",
            [int(self.handles_dict['octree'])],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            print("Error in getting octree voxels: {}".format(res[0]))
        assert len(res[2]) > 0, "Empty octree voxels"
        return res[2]

    def update_octree(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_UpdateOctreeVoxels",
            [int(self.handles_dict['octree']),
             int(self.handles_dict['anchor_obj']),
             int(self.handles_dict['other_obj'])],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            print("Error in udpate octree: {}".format(res[0]))

    def clear_octree(self):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_ClearOctree",
            [int(self.handles_dict['octree'])],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
        if res[0] != vrep.simx_return_ok:
            print("Error in clear octree: {}".format(res[0]))

    def apply_force(self, position, force):
        res = vrep.simxCallScriptFunction(
            self.client_id,
            "Cuboid",
            vrep.sim_scripttype_childscript,
            "ms_ApplyForce",
            [int(self.handles_dict['other_obj'])],
            [position[0], position[1], position[2],
             force[0], force[1], force[2]],
            [],
            bytearray(),
            vrep.simx_opmode_blocking
        )
        if res[0] != vrep.simx_return_ok:
            print("Error in apply force: {}".format(res[0]))

    def sim_relation(self, sim_count):
        anchor_handle = self.handles_dict['anchor_obj']
        other_handle = self.handles_dict['other_obj']

        lb, ub = self.get_object_bb(anchor_handle)
        # print("lb: {}, ub: {}".format(lb, ub))
        # low_x, low_y = lb[0] + 0.3, lb[1] + 0.3
        # high_x, high_y = ub[0] - 0.3, ub[1] - 0.3
        # print("Low: ({:.4f}, {:.4f}), high: ({:4f}, {:.4f})".format(
        #     low_x, low_y, high_x, high_y))

    def run_scene(self, i):
        print("Start sim: {}".format(i))
        scene.reset()

        scene.sim_relation(0)

        # anchor_bb = scene.get_object_bb(scene.handles_dict['anchor_obj'])
        # anchor_pos = scene.get_object_position(scene.handles_dict['anchor_obj'])
        anchor_abs_bb = scene.get_absolute_bb(scene.handles_dict['anchor_obj'])
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        print("Abs bb: \n"
              "\t    lb: \t  {}\n"
              "\t    ub: \t  {}\n".format(anchor_lb, anchor_ub))
        other_obj_z = anchor_ub[2] + other_obj_ht

        low_x = anchor_lb[0] + 0.01
        low_y = anchor_lb[1] + 0.01
        high_x = anchor_ub[0] - 0.01
        high_y = anchor_ub[1] - 0.01
        print("low: ({},{}), high: ({},{})".format(
            low_x, low_y, high_x, high_y))

        if i % 5 == 0 or True:
            other_obj_x = npu(low_x, low_x+0.02)
            other_obj_y = npu(low_y, high_y)
        elif i % 5 == 1:
            other_obj_x = npu(high_x-0.02, high_x)
            other_obj_y = npu(low_y, high_y)
        elif i % 5 == 2: 
            other_obj_x = npu(low_x, high_x)
            other_obj_y = npu(low_y, low_y+0.02)
        elif i % 5 == 3:
            other_obj_x = npu(low_x, high_x)
            other_obj_y = npu(high_y-0.02, high_y)
        else:
            other_obj_x = npu(low_x+0.02, high_x-0.02)
            other_obj_y = npu(low_y+0.02, high_y-0.02)

        other_obj_angles = scene.get_object_orientation(
            scene.handles_dict['other_obj'])

        actions = [
            (-15,0), (-12,0), (-10,0), (-8,0), (8,0), (10,0), (12,0), (15,0),
            (0,-15), (0,-12), (0,-10), (0,-8), (0,8), (0,10), (0,12), (0,15)
        ]

        for a in actions:
                scene.set_object_position(
                    scene.handles_dict['other_obj'],
                    [other_obj_x, other_obj_y, other_obj_z]
                )
                scene.set_object_orientation(
                    scene.handles_dict['other_obj'],
                    other_obj_angles
                )
                # Wait for sometime
                for t in range(50):
                    scene.step()

                other_lb, other_ub = scene.get_absolute_bb(
                    scene.handles_dict['other_obj'])
                print("Initial Abs bb: \n"
                      "\t    lb: \t  {}\n"
                      "\t    ub: \t  {}\n".format(other_lb, other_ub))

                # Get octree representation
                scene.update_octree()
                for t in range(10):
                    scene.step()
                voxels = scene.get_octree_voxels()
                print("Got initial voxels: {}".format(len(voxels)))

                scene.clear_octree()
                for t in range(10):
                    scene.step()

                # push
                scene.apply_force([0, 0, 0], [a[0], a[1], 0])

                # wait and observe effect
                for t in range(50):
                    scene.step()

                # Get final pose.
                other_lb, other_ub = scene.get_absolute_bb(
                    scene.handles_dict['other_obj'])
                print("Final Abs bb: \n"
                    "\t    lb: \t  {}\n"
                    "\t    ub: \t  {}\n".format(other_lb, other_ub))

                for t in range(5):
                    scene.step()

                print(" ==== Scene {} Done ====".format(i))


def main(args):
    scene = VrepScene(args)
    other_obj_ht = 0.1
    other_obj_size = [0.1, 0.1, 0.1]
    for i in range(10):
        print("Start sim: {}".format(i))
        scene.reset()

        scene.sim_relation(0)

        # anchor_bb = scene.get_object_bb(scene.handles_dict['anchor_obj'])
        # anchor_pos = scene.get_object_position(scene.handles_dict['anchor_obj'])
        anchor_abs_bb = scene.get_absolute_bb(scene.handles_dict['anchor_obj'])
        anchor_lb, anchor_ub = anchor_abs_bb[0], anchor_abs_bb[1]
        print("Abs bb: \n"
              "\t    lb: \t  {}\n"
              "\t    ub: \t  {}\n".format(anchor_lb, anchor_ub))
        other_obj_z = anchor_ub[2] + other_obj_ht

        low_x = anchor_lb[0] + 0.01
        low_y = anchor_lb[1] + 0.01
        high_x = anchor_ub[0] - 0.01
        high_y = anchor_ub[1] - 0.01
        print("low: ({},{}), high: ({},{})".format(
            low_x, low_y, high_x, high_y))

        if i % 5 == 0 or True:
            other_obj_x = npu(low_x, low_x+0.02)
            other_obj_y = npu(low_y, high_y)
        elif i % 5 == 1:
            other_obj_x = npu(high_x-0.02, high_x)
            other_obj_y = npu(low_y, high_y)
        elif i % 5 == 2: 
            other_obj_x = npu(low_x, high_x)
            other_obj_y = npu(low_y, low_y+0.02)
        elif i % 5 == 3:
            other_obj_x = npu(low_x, high_x)
            other_obj_y = npu(high_y-0.02, high_y)
        else:
            other_obj_x = npu(low_x+0.02, high_x-0.02)
            other_obj_y = npu(low_y+0.02, high_y-0.02)

        other_obj_angles = scene.get_object_orientation(
            scene.handles_dict['other_obj'])

        actions = [
            (-15,0), (-12,0), (-10,0), (-8,0), (8,0), (10,0), (12,0), (15,0),
            (0,-15), (0,-12), (0,-10), (0,-8), (0,8), (0,10), (0,12), (0,15)
        ]

        for a in actions:
                scene.set_object_position(
                    scene.handles_dict['other_obj'],
                    [other_obj_x, other_obj_y, other_obj_z]
                )
                scene.set_object_orientation(
                    scene.handles_dict['other_obj'],
                    other_obj_angles
                )
                # Wait for sometime
                for t in range(50):
                    scene.step()

                other_lb, other_ub = scene.get_absolute_bb(
                    scene.handles_dict['other_obj'])
                print("Initial Abs bb: \n"
                      "\t    lb: \t  {}\n"
                      "\t    ub: \t  {}\n".format(other_lb, other_ub))

                # Get octree representation
                scene.update_octree()
                for t in range(10):
                    scene.step()
                voxels = scene.get_octree_voxels()
                print("Got initial voxels: {}".format(len(voxels)))

                scene.clear_octree()
                for t in range(10):
                    scene.step()

                # push
                scene.apply_force([0, 0, 0], [a[0], a[1], 0])

                # wait and observe effect
                for t in range(50):
                    scene.step()

                # Get final pose.
                other_lb, other_ub = scene.get_absolute_bb(
                    scene.handles_dict['other_obj'])
                print("Final Abs bb: \n"
                    "\t    lb: \t  {}\n"
                    "\t    ub: \t  {}\n".format(other_lb, other_ub))

                for t in range(5):
                    scene.step()

                print(" ==== Scene {} Done ====".format(i))

    scene.stop_and_finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create on scene in vrep")
    parser.add_argument('--save_dir', type=str, default='/tmp/whatever',
                        help='Save results in dir.')
    parser.add_argument('--scene_file', type=str, required=True,
                        help='Path to scene file.')
    args = parser.parse_args()
    np.set_printoptions(precision=4)
    main(args)
