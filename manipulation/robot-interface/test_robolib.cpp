#include<iostream>

#include <iam_robolib/run_loop.h>
#include <mutex>

int main() {
  std::cout << "Hello world\n";
  std::mutex m;
  std::mutex control_loop_data_mutex;
  std::mutex robot_loop_data_mutex;
  run_loop rl = run_loop(std::ref(m), std::ref(control_loop_data_mutex), std::ref(robot_loop_data_mutex));
  std::cout << "Will start run loop.\n";
  rl.start();
  std::cout << "Did start run loop.\n";
  std::cout << "Will run..\n";
  rl.run_on_franka();
  return 0;
}
