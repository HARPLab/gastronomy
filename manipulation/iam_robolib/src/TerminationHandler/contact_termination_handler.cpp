//
// Created by mohit on 11/26/18.
//

#include "contact_termination_handler.h"

#include <cmath>
#include <iostream>

#include "TrajectoryGenerator/trajectory_generator.h"
#include "TrajectoryGenerator/counter_trajectory_generator.h"

void ContactTerminationHandler::parse_parameters() {
  // First parameter is reserved for the type

  int num_params = static_cast<int>(params_[1]);

  // No collision behavior parameters were provided, so just using default parameters
  // Checking for collisions in any direction or rotation
  if(num_params == 0) 
  {
    lower_torque_thresholds_acceleration_ = default_lower_torque_thresholds_acceleration_;
    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    lower_torque_thresholds_nominal_ = default_lower_torque_thresholds_nominal_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    lower_force_thresholds_acceleration_ = default_lower_force_thresholds_acceleration_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    lower_force_thresholds_nominal_ = default_lower_force_thresholds_nominal_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // Buffer time was given (1) but no collision behavior parameters were provided, so just using default parameters
  // Checking for collisions in any direction or rotation
  else if(num_params == 1) 
  {
    buffer_time_ = static_cast<double>(params_[2]);

    lower_torque_thresholds_acceleration_ = default_lower_torque_thresholds_acceleration_;
    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    lower_torque_thresholds_nominal_ = default_lower_torque_thresholds_nominal_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    lower_force_thresholds_acceleration_ = default_lower_force_thresholds_acceleration_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    lower_force_thresholds_nominal_ = default_lower_force_thresholds_nominal_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // No collision behavior parameters were provided, so just using default parameters
  // Checking for collisions in only the directions or rotations that are nonzero.
  else if(num_params == 6) 
  {
    for(int i = 0; i < 6; i++)
    {
      cartesian_contacts_to_use_[i] = static_cast<double>(params_[2+i]);
    }

    lower_torque_thresholds_acceleration_ = default_lower_torque_thresholds_acceleration_;
    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    lower_torque_thresholds_nominal_ = default_lower_torque_thresholds_nominal_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    lower_force_thresholds_acceleration_ = default_lower_force_thresholds_acceleration_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    lower_force_thresholds_nominal_ = default_lower_force_thresholds_nominal_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // Buffer time was given (1) but no collision behavior parameters were provided, so just using default parameters
  // Checking for collisions in only the directions or rotations that are nonzero.
  else if(num_params == 7) 
  {
    buffer_time_ = static_cast<double>(params_[2]);

    for(int i = 0; i < 6; i++)
    {
      cartesian_contacts_to_use_[i] = static_cast<double>(params_[3+i]);
    }

    lower_torque_thresholds_acceleration_ = default_lower_torque_thresholds_acceleration_;
    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    lower_torque_thresholds_nominal_ = default_lower_torque_thresholds_nominal_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    lower_force_thresholds_acceleration_ = default_lower_force_thresholds_acceleration_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    lower_force_thresholds_nominal_ = default_lower_force_thresholds_nominal_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // New contact thresholds were provided (7,7,6,6)
  // Checking for collisions in any direction or rotation
  else if(num_params == 26) 
  {

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[2+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[9+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[16+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[22+i]);
    }

    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // Buffer time (1) and new contact thresholds were provided (7,7,6,6)
  // Checking for collisions in any direction or rotation
  else if(num_params == 27) 
  {
    buffer_time_ = static_cast<double>(params_[2]);

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[3+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[10+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[17+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[23+i]);
    }

    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // New contact thresholds were provided (7,7,6,6)
  // Checking for collisions in only the directions or rotations that are nonzero.
  else if(num_params == 32) 
  {

    for(int i = 0; i < 6; i++)
    {
      cartesian_contacts_to_use_[i] = static_cast<double>(params_[2+i]);
    }

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[8+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[15+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[22+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[28+i]);
    }

    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // Buffer time (1) and new contact thresholds were provided (7,7,6,6)
  // Checking for collisions in only the directions or rotations that are nonzero.
  else if(num_params == 33) 
  {
    buffer_time_ = static_cast<double>(params_[2]);

    for(int i = 0; i < 6; i++)
    {
      cartesian_contacts_to_use_[i] = static_cast<double>(params_[3+i]);
    }

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[9+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[16+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[23+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[29+i]);
    }

    upper_torque_thresholds_acceleration_ = default_upper_torque_thresholds_acceleration_;
    upper_torque_thresholds_nominal_ = default_upper_torque_thresholds_nominal_;
    upper_force_thresholds_acceleration_ = default_upper_force_thresholds_acceleration_;
    upper_force_thresholds_nominal_ = default_upper_force_thresholds_nominal_;
  } 
  // New contact and collision thresholds were provided (7,7,7,7,6,6,6,6)
  // Checking for collisions in any direction or rotation
  else if(num_params == 52) 
  {
    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[2+i]);
      upper_torque_thresholds_acceleration_[i] = static_cast<double>(params_[9+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[16+i]);
      upper_torque_thresholds_nominal_[i] = static_cast<double>(params_[23+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[30+i]);
      upper_force_thresholds_acceleration_[i] = static_cast<double>(params_[36+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[42+i]);
      upper_force_thresholds_nominal_[i] = static_cast<double>(params_[48+i]);
    }
  } 
  // Buffer time (1) and new contact and collision thresholds were provided (7,7,7,7,6,6,6,6)
  // Checking for collisions in any direction or rotation
  else if(num_params == 53) 
  {
    buffer_time_ = static_cast<double>(params_[2]);

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[3+i]);
      upper_torque_thresholds_acceleration_[i] = static_cast<double>(params_[10+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[17+i]);
      upper_torque_thresholds_nominal_[i] = static_cast<double>(params_[24+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[31+i]);
      upper_force_thresholds_acceleration_[i] = static_cast<double>(params_[37+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[43+i]);
      upper_force_thresholds_nominal_[i] = static_cast<double>(params_[49+i]);
    }
  } 
  // New contact and collision thresholds were provided (7,7,7,7,6,6,6,6)
  // Checking for collisions in any direction or rotation
  else if(num_params == 58) 
  {
    for(int i = 0; i < 6; i++)
    {
      cartesian_contacts_to_use_[i] = static_cast<double>(params_[2+i]);
    }

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[8+i]);
      upper_torque_thresholds_acceleration_[i] = static_cast<double>(params_[15+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[22+i]);
      upper_torque_thresholds_nominal_[i] = static_cast<double>(params_[29+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[36+i]);
      upper_force_thresholds_acceleration_[i] = static_cast<double>(params_[42+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[48+i]);
      upper_force_thresholds_nominal_[i] = static_cast<double>(params_[54+i]);
    }
  } 
  // Buffer time (1) and new contact and collision thresholds were provided (7,7,7,7,6,6,6,6)
  // Checking for collisions in any direction or rotation
  else if(num_params == 59) 
  {
    buffer_time_ = static_cast<double>(params_[2]);

    for(int i = 0; i < 6; i++)
    {
      cartesian_contacts_to_use_[i] = static_cast<double>(params_[3+i]);
    }

    for(int i = 0; i < 7; i++)
    {
      lower_torque_thresholds_acceleration_[i] = static_cast<double>(params_[9+i]);
      upper_torque_thresholds_acceleration_[i] = static_cast<double>(params_[16+i]);
      lower_torque_thresholds_nominal_[i] = static_cast<double>(params_[23+i]);
      upper_torque_thresholds_nominal_[i] = static_cast<double>(params_[30+i]);
    }
    for(int i = 0; i < 6; i++)
    {
      lower_force_thresholds_acceleration_[i] = static_cast<double>(params_[37+i]);
      upper_force_thresholds_acceleration_[i] = static_cast<double>(params_[43+i]);
      lower_force_thresholds_nominal_[i] = static_cast<double>(params_[49+i]);
      upper_force_thresholds_nominal_[i] = static_cast<double>(params_[55+i]);
    }
  } 
  else
  {
    std::cout << "Contact Termination Handler: Invalid number of params provided: " << num_params << std::endl;
  }
}

void ContactTerminationHandler::initialize_handler() {
  // pass
}

void ContactTerminationHandler::initialize_handler(franka::Robot *robot) {
  robot->setCollisionBehavior(lower_torque_thresholds_acceleration_, 
                              upper_torque_thresholds_acceleration_,
                              lower_torque_thresholds_nominal_,
                              upper_torque_thresholds_nominal_,
                              lower_force_thresholds_acceleration_,
                              upper_force_thresholds_acceleration_,
                              lower_force_thresholds_nominal_,
                              upper_force_thresholds_nominal_);
}

bool ContactTerminationHandler::should_terminate(TrajectoryGenerator *trajectory_generator) {
  assert(false);
  return true;
}

bool ContactTerminationHandler::should_terminate(const franka::RobotState &robot_state,
                                                 TrajectoryGenerator *trajectory_generator) {
  if(!done_) {
    if(trajectory_generator->time_ > trajectory_generator->run_time_ + buffer_time_) {
      done_ = true;
      return done_;
    }

    std::array<double, 6> cartesian_contact = robot_state.cartesian_contact;

    for(int i = 0; i < 6; i++) {
      if(cartesian_contacts_to_use_[i] != 0 && cartesian_contact[i] != 0) {
        done_ = true;
        return  done_;
      }
    }
  }
  
  return done_;
}
