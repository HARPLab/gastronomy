cmake_minimum_required (VERSION 3.4)
project (iam_robolib
  VERSION 0.0.1 
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)

## Dependencies
# find_package(Poco REQUIRED COMPONENTS Net Foundation)
find_package(Eigen3 REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(Boost COMPONENTS REQUIRED filesystem system thread)

include_directories(${Boost_INCLUDE_DIRS})

# find_package(nlohmann_json 3.4.0 REQUIRED)

# find_library(iam_robolib_common PATHS ../)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../iam_robolib_common")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../libfranka")

## Library

add_library(iam_robolib SHARED
  src/run_loop.cpp
  src/duration.cpp
  src/TerminationHandler/contact_termination_handler.cpp
  src/TerminationHandler/contact_termination_handler.h
  src/Skills/skill_info.cpp
  src/Skills/skill_info.h
  src/skill_info_manager.cpp
  src/skill_info_manager.h
  src/TrajectoryGenerator/trajectory_generator.cpp
  src/TrajectoryGenerator/trajectory_generator.h
  src/TrajectoryGenerator/goal_linear_trajectory_generator.cpp
  src/TrajectoryGenerator/goal_linear_trajectory_generator.h
  src/TrajectoryGenerator/linear_trajectory_generator.cpp
  src/TrajectoryGenerator/linear_trajectory_generator.h
  src/TrajectoryGenerator/linear_trajectory_generator_with_time_and_goal.cpp
  src/TrajectoryGenerator/linear_trajectory_generator_with_time_and_goal.h
  src/TrajectoryGenerator/linear_trajectory_generator_with_time_and_goal_termination_handler.cpp
  src/TrajectoryGenerator/linear_trajectory_generator_with_time_and_goal_termination_handler.h
  src/FeedbackController/torque_feedback_controller.cpp
  src/FeedbackController/torque_feedback_controller.h
  src/TrajectoryGenerator/counter_trajectory_generator.cpp
  src/TrajectoryGenerator/counter_trajectory_generator.h
  src/FeedbackController/feedback_controller.cpp
  src/FeedbackController/feedback_controller.h
  src/FeedbackController/noop_feedback_controller.cpp
  src/FeedbackController/noop_feedback_controller.h
  src/TerminationHandler/termination_handler.cpp
  src/TerminationHandler/termination_handler.h
  src/TerminationHandler/time_termination_handler.cpp
  src/TerminationHandler/time_termination_handler.h
  src/TrajectoryGenerator/stay_in_initial_position_trajectory_generator.cpp
  src/TrajectoryGenerator/stay_in_initial_position_trajectory_generator.h
  src/sensor_data.cpp
  src/sensor_data.h
  src/TerminationHandler/noop_termination_handler.cpp
  src/TerminationHandler/noop_termination_handler.h
  src/run_loop_logger.cpp
  src/run_loop_logger.h
  src/TerminationHandler/final_pose_termination_handler.cpp
  src/TerminationHandler/final_pose_termination_handler.h
  src/control_loop_data.cpp
  src/control_loop_data.h
  src/TerminationHandler/final_joint_termination_handler.cpp
  src/TerminationHandler/final_joint_termination_handler.h
  src/TrajectoryGenerator/linear_joint_trajectory_controller.cpp
  src/TrajectoryGenerator/linear_joint_trajectory_controller.h
  src/Skills/gripper_open_skill.cpp
  src/Skills/gripper_open_skill.h
  src/TrajectoryGenerator/gripper_open_trajectory_generator.cpp
  src/TrajectoryGenerator/gripper_open_trajectory_generator.h
  src/Skills/base_skill.cpp
  src/Skills/base_skill.h
  src/file_stream_logger.cpp
  src/file_stream_logger.h
  src/TrajectoryGenerator/dmp_trajectory_generator.cpp
  src/TrajectoryGenerator/dmp_trajectory_generator.h
  src/Skills/joint_pose_skill.cpp
  src/Skills/joint_pose_skill.h
  src/Skills/joint_pose_with_torque_control_skill.cpp
  src/Skills/joint_pose_with_torque_control_skill.h
  src/TrajectoryGenerator/relative_linear_trajectory_generator.cpp
  src/TrajectoryGenerator/relative_linear_trajectory_generator.h
  src/FeedbackController/custom_gain_torque_controller.cpp
  src/FeedbackController/custom_gain_torque_controller.h
  src/Skills/save_trajectory_skill.cpp
  src/Skills/save_trajectory_skill.h
  src/Skills/joint_pose_continuous_skill.cpp
  src/Skills/joint_pose_continuous_skill.h
  src/Skills/base_meta_skill.cpp
  src/Skills/base_meta_skill.h
  src/run_loop_shared_memory_handler.cpp
  src/run_loop_shared_memory_handler.h
  src/trajectory_generator_factory.cpp
  src/trajectory_generator_factory.h
  src/feedback_controller_factory.cpp
  src/feedback_controller_factory.h
  src/termination_handler_factory.cpp
  src/termination_handler_factory.h
  src/definitions.h
)

# Add namespace
# add_library(IAM_RoboLib::IAM_RoboLib ALIAS iam_robolib)

target_link_libraries(iam_robolib PRIVATE
  Eigen3::Eigen3
  Threads::Threads
  ${Boost_LIBRARIES}
  rt
  iam_robolib_common
  Franka::Franka
  # nlohmann_json::nlohmann_json
)


target_include_directories(iam_robolib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

set_target_properties(iam_robolib PROPERTIES PUBLIC_HEADER include/run_loop.h)

target_include_directories(iam_robolib PRIVATE src)
target_include_directories(iam_robolib PRIVATE include)

## Installation
include(GNUInstallDirs)
install(TARGETS iam_robolib
  # EXPORT FrankaTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  # RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
