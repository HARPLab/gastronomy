import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='same_blocks/pick_up/spatula_flat/block0_pick_up_block_with_spatula_flat')
    args = parser.parse_args()

    if 'franka_fingers' in args.file or 'tongs_overhead' in args.file:
        x_y_thetas = np.zeros((0,3))
        initial_block_pose = np.zeros((0,7))

        pre_grasp_contact_forces = np.zeros((0,3))
        pre_grasp_robot_pose = np.zeros((0,7))
        desired_pre_grasp_robot_pose = np.zeros((0,7))
        pre_grasp_block_pose = np.zeros((0,7))

        grasp_contact_forces = np.zeros((0,3))
        grasp_block_pose = np.zeros((0,7))

        post_grasp_contact_forces = np.zeros((0,3))
        post_grasp_block_pose = np.zeros((0,7))

        post_release_block_pose = np.zeros((0,7))


        for i in range(1,7):
            data = np.load(args.file+'_'+str(i)+'.npz')
            x_y_thetas = np.vstack((x_y_thetas,data['x_y_thetas']))
            initial_block_pose = np.vstack((initial_block_pose,data['initial_block_pose']))
            pre_grasp_block_pose = np.vstack((pre_grasp_block_pose,data['pre_grasp_block_pose']))
            desired_pre_grasp_robot_pose = np.vstack((desired_pre_grasp_robot_pose,data['desired_pre_grasp_robot_pose']))
            pre_grasp_robot_pose = np.vstack((pre_grasp_robot_pose,data['pre_grasp_robot_pose']))
            pre_grasp_contact_forces = np.vstack((pre_grasp_contact_forces,data['pre_grasp_contact_forces']))
            pre_grasp_block_pose = np.vstack((pre_grasp_block_pose,data['pre_grasp_block_pose']))
            desired_pre_grasp_robot_pose = np.vstack((desired_pre_grasp_robot_pose,data['desired_pre_grasp_robot_pose']))
            pre_grasp_robot_pose = np.vstack((pre_grasp_robot_pose,data['pre_grasp_robot_pose']))
            grasp_contact_forces = np.vstack((grasp_contact_forces,data['grasp_contact_forces']))
            grasp_block_pose = np.vstack((grasp_block_pose,data['grasp_block_pose']))
            post_grasp_contact_forces = np.vstack((post_grasp_contact_forces,data['post_grasp_contact_forces']))
            post_grasp_block_pose = np.vstack((post_grasp_block_pose,data['post_grasp_block_pose']))
            post_release_block_pose = np.vstack((post_release_block_pose,data['post_release_block_pose']))

        np.savez(args.file+'.npz', x_y_thetas=x_y_thetas, 
                                   initial_block_pose=initial_block_pose,
                                   pre_grasp_contact_forces=pre_grasp_contact_forces,
                                   pre_grasp_block_pose=pre_grasp_block_pose,
                                   desired_pre_grasp_robot_pose=desired_pre_grasp_robot_pose,
                                   pre_grasp_robot_pose=pre_grasp_robot_pose,
                                   grasp_contact_forces=grasp_contact_forces,
                                   grasp_block_pose=grasp_block_pose,
                                   post_grasp_contact_forces=post_grasp_contact_forces,
                                   post_grasp_block_pose=post_grasp_block_pose,
                                   post_release_block_pose=post_release_block_pose)

    elif 'tongs_side' in args.file:
        x_y_theta_dist_tilts = np.zeros((0,5))
        initial_block_pose = np.zeros((0,7))

        pre_push_robot_pose = np.zeros((0,7))
        desired_pre_push_robot_pose = np.zeros((0,7))
        pre_push_block_pose = np.zeros((0,7))

        pre_grasp_contact_forces = np.zeros((0,3))
        pre_grasp_robot_pose = np.zeros((0,7))
        desired_pre_grasp_robot_pose = np.zeros((0,7))
        pre_grasp_block_pose = np.zeros((0,7))

        grasp_contact_forces = np.zeros((0,3))
        grasp_block_pose = np.zeros((0,7))

        post_grasp_contact_forces = np.zeros((0,3))
        post_grasp_block_pose = np.zeros((0,7))

        post_release_block_pose = np.zeros((0,7))


        for i in range(1,7):
            data = np.load(args.file+'_'+str(i)+'.npz')
            x_y_theta_dist_tilts = np.vstack((x_y_theta_dist_tilts,data['x_y_theta_dist_tilts']))
            initial_block_pose = np.vstack((initial_block_pose,data['initial_block_pose']))
            pre_push_robot_pose = np.vstack((pre_push_robot_pose,data['pre_push_robot_pose']))
            desired_pre_push_robot_pose = np.vstack((desired_pre_push_robot_pose,data['desired_pre_push_robot_pose']))
            pre_push_block_pose = np.vstack((pre_push_block_pose,data['pre_push_block_pose']))
            pre_grasp_contact_forces = np.vstack((pre_grasp_contact_forces,data['pre_grasp_contact_forces']))
            pre_grasp_block_pose = np.vstack((pre_grasp_block_pose,data['pre_grasp_block_pose']))
            desired_pre_grasp_robot_pose = np.vstack((desired_pre_grasp_robot_pose,data['desired_pre_grasp_robot_pose']))
            pre_grasp_robot_pose = np.vstack((pre_grasp_robot_pose,data['pre_grasp_robot_pose']))
            grasp_contact_forces = np.vstack((grasp_contact_forces,data['grasp_contact_forces']))
            grasp_block_pose = np.vstack((grasp_block_pose,data['grasp_block_pose']))
            post_grasp_contact_forces = np.vstack((post_grasp_contact_forces,data['post_grasp_contact_forces']))
            post_grasp_block_pose = np.vstack((post_grasp_block_pose,data['post_grasp_block_pose']))
            post_release_block_pose = np.vstack((post_release_block_pose,data['post_release_block_pose']))

        np.savez(args.file+'.npz', x_y_theta_dist_tilts=x_y_theta_dist_tilts, 
                                   initial_block_pose=initial_block_pose,
                                   pre_push_robot_pose=pre_push_robot_pose,
                                   desired_pre_push_robot_pose=desired_pre_push_robot_pose,
                                   pre_push_block_pose=pre_push_block_pose,
                                   pre_grasp_contact_forces=pre_grasp_contact_forces,
                                   pre_grasp_block_pose=pre_grasp_block_pose,
                                   desired_pre_grasp_robot_pose=desired_pre_grasp_robot_pose,
                                   pre_grasp_robot_pose=pre_grasp_robot_pose,
                                   grasp_contact_forces=grasp_contact_forces,
                                   grasp_block_pose=grasp_block_pose,
                                   post_grasp_contact_forces=post_grasp_contact_forces,
                                   post_grasp_block_pose=post_grasp_block_pose,
                                   post_release_block_pose=post_release_block_pose)
    elif 'spatula' in args.file:
        if 'tilted' in args.file:
            x_y_theta_dist_tilts = np.zeros((0,5))
        elif 'flat' in args.file: 
            x_y_theta_dists = np.zeros((0,4))

        initial_block_pose = np.zeros((0,7))

        pre_push_block_pose = np.zeros((0,7))
        pre_push_robot_pose = np.zeros((0,7))
        desired_pre_push_robot_pose = np.zeros((0,7))

        push_block_pose = np.zeros((0,7))
        push_robot_pose = np.zeros((0,7))
        desired_push_robot_pose = np.zeros((0,7))
        
        post_pick_up_block_pose = np.zeros((0,7))
        
        post_release_block_pose = np.zeros((0,7))

        for i in range(1,7):
            data = np.load(args.file+'_'+str(i)+'.npz')
            if 'tilted' in args.file:
                x_y_theta_dist_tilts = np.vstack((x_y_theta_dist_tilts,data['x_y_theta_dist_tilts']))
            elif 'flat' in args.file: 
                x_y_theta_dists = np.vstack((x_y_theta_dists,data['x_y_theta_dists']))

            initial_block_pose = np.vstack((initial_block_pose,data['initial_block_pose']))
            pre_push_robot_pose = np.vstack((pre_push_robot_pose,data['pre_push_robot_pose']))
            desired_pre_push_robot_pose = np.vstack((desired_pre_push_robot_pose,data['desired_pre_push_robot_pose']))
            pre_push_block_pose = np.vstack((pre_push_block_pose,data['pre_push_block_pose']))
            push_block_pose = np.vstack((push_block_pose,data['push_block_pose']))
            push_robot_pose = np.vstack((push_robot_pose,data['push_robot_pose']))
            desired_push_robot_pose = np.vstack((desired_push_robot_pose,data['desired_push_robot_pose']))

            post_pick_up_block_pose = np.vstack((post_pick_up_block_pose,data['post_pick_up_block_pose']))
            post_release_block_pose = np.vstack((post_release_block_pose,data['post_release_block_pose']))

        if 'tilted' in args.file:
            np.savez(args.file+'.npz', x_y_theta_dist_tilts=x_y_theta_dist_tilts, 
                                   initial_block_pose=initial_block_pose,
                                   pre_push_robot_pose=pre_push_robot_pose,
                                   desired_pre_push_robot_pose=desired_pre_push_robot_pose,
                                   pre_push_block_pose=pre_push_block_pose,
                                   push_block_pose=push_block_pose,
                                   push_robot_pose=push_robot_pose,
                                   desired_push_robot_pose=desired_push_robot_pose,
                                   post_pick_up_block_pose=post_pick_up_block_pose,
                                   post_release_block_pose=post_release_block_pose)
        elif 'flat' in args.file: 
            np.savez(args.file+'.npz', x_y_theta_dists=x_y_theta_dists, 
                                   initial_block_pose=initial_block_pose,
                                   pre_push_robot_pose=pre_push_robot_pose,
                                   desired_pre_push_robot_pose=desired_pre_push_robot_pose,
                                   pre_push_block_pose=pre_push_block_pose,
                                   push_block_pose=push_block_pose,
                                   push_robot_pose=push_robot_pose,
                                   desired_push_robot_pose=desired_push_robot_pose,
                                   post_pick_up_block_pose=post_pick_up_block_pose,
                                   post_release_block_pose=post_release_block_pose)

        