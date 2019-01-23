cmake_minimum_required (VERSION 3.4)
project (shared_memory_test_app
  VERSION 0.0.1 
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Dependencies
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(Boost COMPONENTS REQUIRED filesystem system thread)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(shared_memory_test_app shared_memory_test_app.cpp)

target_sources(shared_memory_test_app 
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/run_loop_process_info.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/run_loop_process_info.h"
)
    

target_link_libraries(shared_memory_test_app PRIVATE
  Threads::Threads
  ${Boost_LIBRARIES}
  rt
)

install(TARGETS shared_memory_test_app DESTINATION bin)
