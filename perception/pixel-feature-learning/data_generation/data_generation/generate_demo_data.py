import sys
import pybullet as p
import pybullet_data
import numpy as np
import math
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils import spartan_utils

from pyquaternion import Quaternion

def add_frame(T, r=0.1):
    center = T[:3, 3]
    x = T[:3, 0] + center
    y = T[:3, 1] + center
    z = T[:3, 2] + center
    visualShapeIdo = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 0, 0, 1], radius=r)
    visualShapeIdx = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=r)
    visualShapeIdy = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 1, 0, 1], radius=r)
    visualShapeIdz = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 0, 1, 1], radius=r)
    mb = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeIdo, basePosition=center, useMaximalCoordinates=True)
    mb = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeIdx, basePosition=x, useMaximalCoordinates=True)
    mb = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeIdy, basePosition=y, useMaximalCoordinates=True)
    mb = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeIdz, basePosition=z, useMaximalCoordinates=True)


def add_point(pos, r=0.1):
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 0, 0, 1], radius=r)
    mb = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, basePosition=pos, useMaximalCoordinates=True)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def linearDepth(depthSample, zNear, zFar):
    depthSample = 2.0 * depthSample - 1.0
    zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear))
    clipping_distance = zFar - 10
    zLinear[np.where(zLinear > clipping_distance)] = 0 # now we have real distances

    # Scale distances up to be similar to Dense Object Net implementation!
    depth_rescaled = zLinear*1000
    return np.asarray(depth_rescaled, dtype=np.uint16)

def getRandomTargetPos():
    x = (np.random.rand() - 0.5) * 0.15
    y = (np.random.rand() - 0.5) * 0.15
    z = (np.random.rand() - 0.5) * 0.05 - 0.05
    return [x, y, z]

def getViewMatrix(roll, pitch, yaw, targetPosition=[0, 0, 0]):
    distance = 0.4 + np.random.rand() * 0.2# randomize camera-distance to object!

    # TODO: change roll should affect viewMatrix
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=targetPosition,
        distance=distance,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        upAxisIndex=2
    )

    T_camera_world = np.linalg.inv(np.array(viewMatrix).reshape((4,4)).transpose())

    camera_quaternion = spartan_utils.quaternion_from_matrix(T_camera_world)
    position = T_camera_world[:3, 3]

    return viewMatrix, position, camera_quaternion

def transformVector(vec, q):
    vec = q.rotate(vec)
    return vec

def main():
    model_files = [
        '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/Broccoli_obj/Broccoli.obj',
        '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/cabbage_obj/cabbage.obj',
        '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/cheese_01-obj/cheese_01.obj',
        # '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/cheese_03-obj/cheese_03.obj',
        # '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/Cucumber_obj/Cucumber.obj',
        '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/egg_obj/egg.obj',
        '/home/zhouxian/Dropbox/projects/sony_cooking/data/processed/jalapeno_pepper_obj/Jalapeno_Pepper.obj'
    ]
    scales = [0.001, 0.0006, 0.03, 0.18, 0.001]
    # scales = [0.001, 0.0006, 0.03, 0.007, 0.0015, 0.18, 0.001]
    mesh_poses = [
        [0, -0.06, 0],
        [-0.06, 0.06, 0],
        [0.05, 0.06, 0],
        # [0, 0.2, 0],
        # [0, 0, 0],
        [-0.06, 0, 0],
        [0, 0, 0]
    ]
    Eulers = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [np.pi/2, 0, 0],
        [0, 0, 0]
    ]
    physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
    output_root_dir = "/home/zhouxian/git/pytorch-dense-correspondence/pdc/logs_proto/cooking/"

    IMG_H = 480
    IMG_W = 640

    nearPlane = 0.01
    farPlane = 100
    fov = 60
    aspect = 1.0 * IMG_W / IMG_H
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

    for i, model_file in enumerate(model_files):
        model_name = os.path.splitext(os.path.split(model_file)[-1])[0]
        mesh_pos = mesh_poses[i]
        mesh_scale=[scales[i]] * 3
        mesh_orientation = p.getQuaternionFromEuler(Eulers[i])
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=model_file, visualFramePosition=mesh_pos, meshScale=mesh_scale, visualFrameOrientation=mesh_orientation)
        p.createMultiBody(baseMass=1, baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=-1, baseVisualShapeIndex = visualShapeId, basePosition = mesh_pos, useMaximalCoordinates=True)
    # import IPython;IPython.embed()

    # taking pictures
    depth_data = dict()
    depth_data['msgs'] = []
    depth_data['cv_img'] = []
    depth_data['timestamps'] = []
    depth_data['camera_to_world'] = []

    rgba_images = []
    depth_images = []
    mask_images = []

    pitch_step = 20
    yaw_step = 10
    roll = 0.
    for pitch in range(-60, -120 , -pitch_step):
        for yaw in range(0, 360, yaw_step):
            viewMatrix, position, camera_quaternion = getViewMatrix(roll, pitch, yaw, getRandomTargetPos())
            # TODO: lightDirection, lightColor not working
            img_as_array_tiny = p.getCameraImage(IMG_W, IMG_H, viewMatrix, projectionMatrix, shadow=1,
                                            lightDirection=[1, 1, 1], renderer=p.ER_TINY_RENDERER)
            img_as_array_GL = p.getCameraImage(IMG_W, IMG_H, viewMatrix, projectionMatrix, shadow=1,
                                            lightDirection=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)

            rgba_img = img_as_array_GL[2]
            depth_img = img_as_array_GL[3]
            depth_img = linearDepth(depth_img, nearPlane, farPlane)
            mask_img = np.abs(img_as_array_tiny[4]+1)

            rgba_images.append(rgba_img)
            depth_images.append(depth_img)
            mask_images.append(mask_img)

            # depth_data['camera_to_world'].append((position.tolist(), camera_quaternion.q.tolist()))
            depth_data['camera_to_world'].append((position.tolist(), camera_quaternion.tolist()))
            depth_data['timestamps'].append(00000000)  # nano seconds (probably not really needed, thats why I put a default value inside here)
            # add_frame(np.array(viewMatrix).reshape((4, 4)).transpose(), 0.1)
            trans = position.tolist()
            # rot = camera_quaternion.q.tolist()
            # quat_wxyz = [rot[3], rot[0], rot[1], rot[2]]
            quat_wxyz = camera_quaternion.tolist()
            transform_dict = spartan_utils.dictFromPosQuat(trans, quat_wxyz)
            T = spartan_utils.homogenous_transform_from_dict(transform_dict)
            # add_frame(T)
            p.stepSimulation() # step simulation for getting next camera view

    pose_data = dict()
    output_dir = os.path.join(output_root_dir, 'cooking_scene', 'processed')
    mask_dir = "image_masks/"
    depth_dir = "rendered_images/"
    rgb_dir = "images/"


    for idx in range(len(depth_data['timestamps'])):

        rgb_filename = "%06i_%s.png" % (idx, "rgb")
        rgb_filename_full = os.path.join(output_dir, rgb_dir, rgb_filename)

        depth_filename = "%06i_%s.png" % (idx, "depth")
        depth_filename_full = os.path.join(output_dir, depth_dir, depth_filename)

        mask_filename = "%06i_%s.png" % (idx, "mask")
        mask_filename_full = os.path.join(output_dir, mask_dir, mask_filename)

        pose_data[idx] = dict()
        d = pose_data[idx]
        trans, quat_wxyz = depth_data['camera_to_world'][idx]
        transform_dict = spartan_utils.dictFromPosQuat(trans, quat_wxyz)
        d['camera_to_world'] = transform_dict
        d['timestamp'] = depth_data['timestamps'][idx]
        d['rgb_image_filename'] = rgb_filename
        d['depth_image_filename'] = depth_filename

        #####
        rgba_img = rgba_images[idx]
        depth_img = depth_images[idx]
        mask_img = mask_images[idx]

        if idx == 0:
            ensure_dir(rgb_filename_full)
            ensure_dir(depth_filename_full)
            ensure_dir(mask_filename_full)
        cv2.imwrite(rgb_filename_full, cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR))
        cv2.imwrite(depth_filename_full, depth_img)
        cv2.imwrite(mask_filename_full, mask_img)

    # Pose-data output. camera_info.yaml has to be generated separately!
    spartan_utils.saveToYaml(pose_data, os.path.join(output_dir, rgb_dir, 'pose_data.yaml'))
    p.resetSimulation()
    p.disconnect()

if __name__ == '__main__':
    main()