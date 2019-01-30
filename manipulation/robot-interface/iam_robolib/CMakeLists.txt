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

include_directories(include)

## Library

add_library(iam_robolib SHARED
  src/run_loop.cpp
  src/duration.cpp
  src/termination_handler/contact_termination_handler.cpp
  src/skills/skill_info.cpp
  src/skill_info_manager.cpp
  src/trajectory_generator/trajectory_generator.cpp
  src/trajectory_generator/linear_trajectory_generator.cpp
  src/trajectory_generator/linear_trajectory_generator_with_time_and_goal.cpp
  src/trajectory_generator/impulse_trajectory_generator.cpp
  src/termination_handler/linear_trajectory_generator_with_time_and_goal_termination_handler.cpp
  src/feedback_controller/torque_feedback_controller.cpp
  src/trajectory_generator/counter_trajectory_generator.cpp
  src/feedback_controller/feedback_controller.cpp
  src/feedback_controller/noop_feedback_controller.cpp
  src/termination_handler/termination_handler.cpp
  src/termination_handler/time_termination_handler.cpp
  src/trajectory_generator/stay_in_initial_position_trajectory_generator.cpp
  src/sensor_data.cpp
  src/termination_handler/noop_termination_handler.cpp
  src/run_loop_logger.cpp
  src/termination_handler/final_pose_termination_handler.cpp
  src/robot_state_data.cpp
  src/termination_handler/final_joint_termination_handler.cpp
  src/trajectory_generator/linear_joint_trajectory_generator.cpp
  src/skills/gripper_open_skill.cpp
  src/trajectory_generator/gripper_open_trajectory_generator.cpp
  src/skills/base_skill.cpp
  src/file_stream_logger.cpp
  src/trajectory_generator/dmp_trajectory_generator.cpp
  src/skills/joint_pose_skill.cpp
  src/skills/joint_pose_with_torque_control_skill.cpp
  src/skills/force_torque_skill.cpp
  src/trajectory_generator/relative_linear_trajectory_generator.cpp
  src/feedback_controller/custom_gain_torque_controller.cpp
  src/skills/save_trajectory_skill.cpp
  src/skills/joint_pose_continuous_skill.cpp
  src/skills/base_meta_skill.cpp
  src/run_loop_shared_memory_handler.cpp
  src/trajectory_generator_factory.cpp
  src/feedback_controller_factory.cpp
  src/termination_handler_factory.cpp
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
