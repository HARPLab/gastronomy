cmake_minimum_required(VERSION 3.12)
project(pcl_segmentation)

find_package(PCL 1.2 REQUIRED)

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_STANDARD 11)

option(EXEC_TYPE "EXEC TYPE" OFF)

# add_executable(pcl_utils main.cpp)
# add_executable(pcl_segmentation main_pre_segmented.cpp)
add_executable(pcl_segmentation main_kmeans_seg.cpp)
target_link_libraries (pcl_segmentation
        ${PCL_LIBRARIES}
        ${Boost_LIBRARIES})
