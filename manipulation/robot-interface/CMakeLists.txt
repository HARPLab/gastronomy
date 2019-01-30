cmake_minimum_required (VERSION 3.4)
project (main_iam_robolib
  VERSION 0.0.1 
  LANGUAGES CXX
)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)

find_package(Eigen3 REQUIRED)

add_subdirectory(iam_robolib)
add_subdirectory(iam_robolib_common)
add_subdirectory(libfranka)
# add_subdirectory(src/catkin_ws/src/franka_action_lib)

add_executable(main_iam_robolib main_iam_robolib.cpp)

target_link_libraries(main_iam_robolib 
  iam_robolib 
  iam_robolib_common
  Franka::Franka
  Eigen3::Eigen3
)

install(TARGETS main_iam_robolib DESTINATION bin)
