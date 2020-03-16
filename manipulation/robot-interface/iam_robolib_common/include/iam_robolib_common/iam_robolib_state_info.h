#pragma once

#include <cassert>
#include <string>

class IAMRobolibStateInfo {
 public:
  IAMRobolibStateInfo(){};

  bool get_is_ready();
  void set_is_ready(bool is_ready);

  int get_watchdog_counter();
  void reset_watchdog_counter(); // to be used by robolib
  void increment_watchdog_counter(); // to be used by franka_actionlib

  std::string get_error_description();
  void set_error_description(std::string description); // to be used by robolib
  void clear_error_description(); // to be used by franka_actionlib

 private:
  bool is_ready_{false};
  int watchdog_counter_{0};
  char error_description_[1000];
  size_t error_description_len_=0;
};

