import numpy as np
from autolab_core import RigidTransform

def get_random_relative_pose(num_iterations):
    relative_transforms = {}

    section_size = int(np.ceil(num_iterations / 16))

    for iter_num in range(num_iterations):
        # Random x, y
        if iter_num < section_size:
            x = (np.random.random() * 2 - 1) * 0.1
            y = (np.random.random() * 2 - 1) * 0.1
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            from_frame='ee_frame', to_frame='ee_frame')
            
        # Random x, y, rx
        elif iter_num < 2 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], 0, 0]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, ry
        elif iter_num < 3 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([0, 0, -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, rz
        elif iter_num < 4 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([0, rz[i], 0]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, rx, ry
        elif iter_num < 5 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], 0, -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, rx, rz
        elif iter_num < 6 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], rz[i], 0]),
                                                            from_frame='ee_frame', to_frame='ee_frame')
        # Random x, y, ry, rz
        elif iter_num < 7 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([0, rz[i], -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, rx, ry, rz
        elif iter_num < 8 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], 0, -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], rz[i], -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')
        # Random x, y, z
        if iter_num < 9 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            from_frame='ee_frame', to_frame='ee_frame')
            
        # Random x, y, z, rx
        elif iter_num < 10 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], 0, 0]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, z, ry
        elif iter_num < 11 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([0, 0, -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, z, rz
        elif iter_num < 12 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([0, rz[i], 0]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, z, rx, ry
        elif iter_num < 13 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], 0, -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

        # Random x, y, z, rx, rz
        elif iter_num < 14 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], rz[i], 0]),
                                                            from_frame='ee_frame', to_frame='ee_frame')
        # Random x, y, z, ry, rz
        elif iter_num < 15 * section_size:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([0, rz[i], -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')
        
        # Random x, y, z, rx, ry, rz
        else:
            x = (np.random.random(num_envs) * 2 - 1) * 0.1
            y = (np.random.random(num_envs) * 2 - 1) * 0.1
            z = (np.random.random(num_envs) * 2 - 1) * 0.1
            rx = (np.random.random(num_envs) * 2 - 1) * np.pi
            ry = (np.random.random(num_envs) * 2 - 1) * np.pi
            rz = (np.random.random(num_envs) * 2 - 1) * np.pi
            relative_transforms[iter_num] = RigidTransform(translation=[x[i], z[i], -y[i]], 
                                                            rotation=RigidTransform.quaternion_from_axis_angle([rx[i], rz[i], -ry[i]]),
                                                            from_frame='ee_frame', to_frame='ee_frame')

    return relative_transforms 

def get_random_x_y_thetas(num_iterations):
    random_x_y_thetas = np.random.random((num_iterations, 3)) * 2.0 - 1.0
    random_x_y_thetas[:,:2] *= 0.1
    random_x_y_thetas[:,2] *= np.pi

    return random_x_y_thetas

def get_random_x_y_z_thetas(num_iterations):
    random_x_y_z_thetas = np.random.random((num_iterations, 4)) * 2.0 - 1.0
    random_x_y_z_thetas[:,:2] *= 0.1
    random_x_y_z_thetas[:,2] = random_x_y_z_thetas[:,2] * 0.1 + 0.2
    random_x_y_z_thetas[:,3] *= np.pi

    return random_x_y_z_thetas

def get_random_x_y_z_theta_dists(num_iterations):
    random_x_y_z_theta_dists = np.random.random((num_iterations, 5))
    random_x_y_z_theta_dists[:,0] = 0.1 * ((random_x_y_z_theta_dists[:,0] * 2.0) - 1.0)
    random_x_y_z_theta_dists[:,1] = 0.1*random_x_y_z_theta_dists[:,1]
    random_x_y_z_theta_dists[:,2] = 0.2 * random_x_y_z_theta_dists[:,2] + 0.1
    random_x_y_z_theta_dists[:,3] = ((random_x_y_z_theta_dists[:,3] * 2.0) - 1.0) * np.pi
    random_x_y_z_theta_dists[:,4] = 0.2 * random_x_y_z_theta_dists[:,4] + 0.2

    return random_x_y_z_theta_dists

def get_random_x_y_theta_dists(num_iterations):
    random_x_y_z_theta_dists = np.random.random((num_iterations, 4))
    random_x_y_z_theta_dists[:,0] = 0.1 * ((random_x_y_z_theta_dists[:,0] * 2.0) - 1.0)
    random_x_y_z_theta_dists[:,1] = 0.1*random_x_y_z_theta_dists[:,1]
    random_x_y_z_theta_dists[:,2] = ((random_x_y_z_theta_dists[:,2] * 2.0) - 1.0) * np.pi
    random_x_y_z_theta_dists[:,3] = 0.2 * random_x_y_z_theta_dists[:,3] + 0.2

    return random_x_y_z_theta_dists

def get_random_x_y_theta_tilt_dists(num_iterations):
    random_x_y_theta_tilt_dists = np.random.random((num_iterations, 5))
    random_x_y_theta_tilt_dists[:,0] = 0.1 * ((random_x_y_theta_tilt_dists[:,0] * 2.0) - 1.0)
    random_x_y_theta_tilt_dists[:,1] = 0.1*random_x_y_theta_tilt_dists[:,1]
    random_x_y_theta_tilt_dists[:,2] = ((random_x_y_theta_tilt_dists[:,2] * 2.0) - 1.0) * np.pi
    random_x_y_theta_tilt_dists[:,3] *= np.pi/2
    random_x_y_theta_tilt_dists[:,4] = 0.1 * random_x_y_theta_tilt_dists[:,4] + 0.2

    return random_x_y_theta_tilt_dists

def get_random_x_y_theta_dist_tilts(num_iterations):
    random_x_y_theta_tilt_dists = np.random.random((num_iterations, 5))
    random_x_y_theta_tilt_dists[:,0] = 0.1 * ((random_x_y_theta_tilt_dists[:,0] * 2.0) - 1.0)
    random_x_y_theta_tilt_dists[:,1] = 0.1*random_x_y_theta_tilt_dists[:,1]
    random_x_y_theta_tilt_dists[:,2] = ((random_x_y_theta_tilt_dists[:,2] * 2.0) - 1.0) * np.pi
    random_x_y_theta_tilt_dists[:,3] = 0.1 * random_x_y_theta_tilt_dists[:,3] + 0.2
    random_x_y_theta_tilt_dists[:,4] *= np.pi/2
    
    return random_x_y_theta_tilt_dists

def get_random_x_y_z_theta_tilt_dists(num_iterations):
    random_x_y_z_theta_tilt_dists = np.random.random((num_iterations, 6))
    random_x_y_z_theta_tilt_dists[:,0] = 0.1 * ((random_x_y_z_theta_tilt_dists[:,0] * 2.0) - 1.0)
    random_x_y_z_theta_tilt_dists[:,1] = 0.1*random_x_y_z_theta_tilt_dists[:,1]
    random_x_y_z_theta_tilt_dists[:,2] = 0.2 * random_x_y_z_theta_tilt_dists[:,2] + 0.1
    random_x_y_z_theta_tilt_dists[:,3] = ((random_x_y_z_theta_tilt_dists[:,3] * 2.0) - 1.0) * np.pi
    random_x_y_z_theta_tilt_dists[:,4] *= np.pi/2
    random_x_y_z_theta_tilt_dists[:,5] = 0.1 * random_x_y_z_theta_tilt_dists[:,5] + 0.2

    return random_x_y_z_theta_tilt_dists

def get_random_x_y_z_rx_ry_rz(num_iterations):
    random_x_y_z_rx_ry_rz = np.random.random((num_iterations, 6)) * 2.0 - 1.0

    section_size = int(np.ceil(num_iterations / 16))

    # x, y, z range between -0.1 and 0.1
    random_x_y_z_rx_ry_rz[:,:3] *= 0.1
    # z is non zero when there are more than 8 * section_size iterations
    random_x_y_z_rx_ry_rz[:(8*section_size),2] = 0.0
    # rx, ry, rz range between -pi and pi
    random_x_y_z_rx_ry_rz[:,3:] *= np.pi
    # First section_size iterations have no orientation changes
    random_x_y_z_rx_ry_rz[:section_size,3:] = 0.0
    # section_size - (2*section_size) only have rx
    random_x_y_z_rx_ry_rz[section_size:(2*section_size),4:] = 0.0
    # (2*section_size) - (3*section_size) only have ry
    random_x_y_z_rx_ry_rz[(2*section_size):(3*section_size),3] = 0.0
    random_x_y_z_rx_ry_rz[(2*section_size):(3*section_size),5] = 0.0
    # (3*section_size) - (4*section_size) only have rz
    random_x_y_z_rx_ry_rz[(3*section_size):(4*section_size),3:5] = 0.0
    # (4*section_size) - (5*section_size) have rx, ry
    random_x_y_z_rx_ry_rz[(4*section_size):(5*section_size),5] = 0.0
    # (5*section_size) - (6*section_size) have rx, rz
    random_x_y_z_rx_ry_rz[(5*section_size):(6*section_size),4] = 0.0
    # (6*section_size) - (7*section_size) have ry, rz
    random_x_y_z_rx_ry_rz[(6*section_size):(7*section_size),3] = 0.0
    # (8*section_size) - (9*section_size) have no orientation changes
    random_x_y_z_rx_ry_rz[(8*section_size):(9*section_size),3:] = 0.0
    # (9*section_size) - (10*section_size) only have rx
    random_x_y_z_rx_ry_rz[(9*section_size):(10*section_size),4:] = 0.0
    # (10*section_size) - (11*section_size) only have ry
    random_x_y_z_rx_ry_rz[(10*section_size):(11*section_size),3] = 0.0
    random_x_y_z_rx_ry_rz[(10*section_size):(11*section_size),5] = 0.0
    # (11*section_size) - (12*section_size) only have rz
    random_x_y_z_rx_ry_rz[(11*section_size):(12*section_size),3:5] = 0.0
    # (12*section_size) - (13*section_size) have rx, ry
    random_x_y_z_rx_ry_rz[(12*section_size):(13*section_size),5] = 0.0
    # (13*section_size) - (14*section_size) have rx, rz
    random_x_y_z_rx_ry_rz[(13*section_size):(14*section_size),4] = 0.0
    # (14*section_size) - (15*section_size) have ry, rz
    random_x_y_z_rx_ry_rz[(14*section_size):(15*section_size),3] = 0.0

    return random_x_y_z_rx_ry_rz

def get_random_rotation_with_3_choices(num_iterations):
    random_numbers = np.random.random((num_iterations,1))
    random_numbers[np.nonzero(random_numbers < 1.0/3)] = -np.pi/2
    random_numbers[np.nonzero(np.logical_and(random_numbers >= 1.0/3,random_numbers < 2.0/3))] = 0
    random_numbers[np.nonzero(random_numbers >= 2.0/3)] = np.pi/2

    return random_numbers