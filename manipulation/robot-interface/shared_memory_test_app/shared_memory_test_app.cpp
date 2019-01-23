#include <iostream>
#include <thread>
#include <array>
#include <chrono>
#include <cassert>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include "run_loop_process_info.h"

typedef float *SharedBuffer;
// using SharedBuffer = std::array<float, 1024>;

struct Task {
  float start_;
  float end_;
  float delta_;

  Task(float start, float end, float delta) : start_(start), end_(end), delta_(delta) {};

  void write_task_params_to_buffer(SharedBuffer buffer) {
    // first index is the task type. Default to 1.
    buffer[0] = 1.0f;
    buffer[1] = start_;
    buffer[2] = end_;
    buffer[3] = delta_;
    std::cout << "Did write to memory "<< buffer[0] << " " << buffer[1] << " "  << buffer[2] << " "
        << buffer[3] << "\n";
  }
};

int main() {
  // TODO(Mohit): All these shared memory methods throw exceptions which we should
  // be catching.


  std::cout << "Hello world\n";
  boost::interprocess::managed_shared_memory segment(
      boost::interprocess::open_only,
      "run_loop_shared_memory"); //Shared memory object name

  // Get RunLoopProcessInfo from the the shared memory segment.
  std::pair<RunLoopProcessInfo*, std::size_t> res = segment.find<RunLoopProcessInfo>
    ("run_loop_info");
  RunLoopProcessInfo *run_loop_process_info = res.first;
  // Make sure the process info object can be found in memory.
  assert(run_loop_process_info != 0);

  // Get mutex for ProcessInfo from the shared memory segment.
  std::pair<boost::interprocess::interprocess_mutex *, std::size_t> mutex_pair = \
      segment.find<boost::interprocess::interprocess_mutex>
          ("run_loop_info_mutex");
  boost::interprocess::interprocess_mutex *run_loop_info_mutex = mutex_pair.first;
  assert(run_loop_info_mutex != 0);


  // Get shared memory object 0
  boost::interprocess::shared_memory_object shm_0(
          boost::interprocess::open_only,
          "run_loop_shared_obj_0",
          boost::interprocess::read_write
  );
  boost::interprocess::mapped_region region_0(shm_0, boost::interprocess::read_write);
  SharedBuffer buffer_0 = reinterpret_cast<SharedBuffer>(region_0.get_address());

  // Get shared memory object 1
  std::string shm_name = run_loop_process_info->get_current_shared_memory_name();
  boost::interprocess::shared_memory_object shm_1(
          boost::interprocess::open_only,
          "run_loop_shared_obj_1",
          boost::interprocess::read_write
  );
  boost::interprocess::mapped_region region_1(shm_1, boost::interprocess::read_write);
  SharedBuffer buffer_1 = reinterpret_cast<SharedBuffer>(region_1.get_address());

  // Now we will have new tasks coming in every 'x' seconds. Each task is a counter
  // based trajectory generator. We count up and down alternatively, beginning from 0, i.e.
  // Task 1: (start: 0, end: 10, delta: 1), Task 2: (start: 0, end: -10, delta: -1)
  // Task 3: (start: 10, end: 20, delta: 1), Task 4: (start: -10, end: -20, delta: -1)
  // Each of these are written alternatively to different shared memory objects.

  Task up_count_task = Task(0, 10, 1);
  Task down_count_task = Task(0, -10, -1);

  int t = 0;
  int current_task_id = 1;
  bool current_task_up = true;
  // Wait for some time
  std::this_thread::sleep_for(std::chrono::seconds(2));
  while (1) {
    // Now we have our task. Try to send it to the shared memory buffer.
    // Use scoped locks
    {
      boost::interprocess::scoped_lock<
                    boost::interprocess::interprocess_mutex> lock(
                        *run_loop_info_mutex,
                        boost::interprocess::defer_lock);
      Task current_task(0, 0, 0);
      if (current_task_up) {
        current_task = up_count_task;
      } else {
        current_task = down_count_task;
      }

      assert(lock.mutex() != 0);
      std::cout << "will try to get lock\n";
      try {
        // TODO(Mohit): Should we use a timed locked here?  This will block forever otherwise.
        lock.lock();

        if (run_loop_process_info->new_task_available_) {
          std::cout << "New task still available. Should not overwrite. Throw exception?"<< std::endl;
        } else {
          int memory_index = run_loop_process_info->get_current_shared_memory_index();
          int actionlib_memory_index = 1 - memory_index;
          std::cout << "Did get lock. Will write to memory: " << actionlib_memory_index << "\n";

          SharedBuffer buffer = buffer_0;
          if (actionlib_memory_index == 1) {
            buffer = buffer_1;
          }

          // Reset the memory.
          // buffer.fill(-1.0);
          std::memset(buffer, -1, sizeof(float) * 1024);
          current_task.write_task_params_to_buffer(buffer);
          run_loop_process_info->update_new_skill(current_task_id);
          run_loop_process_info->new_task_available_ = true;

          std::cout << "Did write task: " << current_task_id << " to shared buffer.\n";

          // Update to new task
          int old_start = current_task.start_;
          int new_start = current_task.end_;
          int old_end = current_task.end_;
          int new_end = old_end + (old_end - old_start);
          current_task.start_ = new_start;
          current_task.end_ = new_end;
          current_task_up = (!current_task_up);
          current_task_id = current_task_id + 1;
        }

      } catch (boost::interprocess::lock_exception) {
        // TODO(Mohit): Do something better here.
        std::cout << "Cannot acquire lock for run loop info\n";
      }
    }

    // sleep for sometime before new task begins.
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  std::cout << "Will exit the shared_memory_test_app" << std::endl;
  return 0;
}
