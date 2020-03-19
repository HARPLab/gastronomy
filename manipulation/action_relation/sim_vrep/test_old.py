import numpy as np

from utilities import vrep_utils as vu

from lib import vrep

vrep.simxFinish(-1)  # just in case, close all opened connections

client_id = vu.connect_to_vrep()
print(client_id)

res = vu.load_scene(
    client_id,
    '/home/mohit/projects/cooking/code/sim_vrep/scene_0.ttt')
print(res)
if res != vrep.simx_return_ok:
    print("Error in loading scene: {}".format(res))

vu.start_sim(client_id)

cam_rgb_names = ["Camera"]
cam_rgb_handles = [vrep.simxGetObjectHandle(
    client_id, n, vrep.simx_opmode_blocking)[1] for n in cam_rgb_names]
print("Camera handle: {}".format(cam_rgb_handles))

octree_handle = vu.get_handle_by_name(client_id, 'Octree')
print('Octree handle: {}'.format(octree_handle))


voxel_list = []
for t in range(100):
    vu.step_sim(client_id)
    res = \
        vrep.simxCallScriptFunction(
            client_id,
            "Octree",
            vrep.sim_scripttype_childscript,
            "ms_GetOctreeVoxels",
            [int(octree_handle)],
            [],
            [],
            bytearray(),
            vrep.simx_opmode_blocking,
        )
    voxels = res[2]
    voxel_list.append(np.array(voxels))
    if t == 50:
        print("diff: {}".format(
            np.sum(np.abs(voxel_list[-1] - voxel_list[5]))))

# removeVoxelsFromOctree(handle, 0, NULL, 0, NULL)  # Removes all voxels
# simInsertObjectIntoOctree(handle, obj_handle, 0, <color>, int_tag, NULL)
vu.stop_sim(client_id)
vrep.simxFinish(client_id)
