#include <iam_robolib_common/iam_robolib_state_info.h>

#include <cstring>

bool IAMRobolibStateInfo::get_is_ready() {
  return is_ready_;
}

void IAMRobolibStateInfo::set_is_ready(bool is_ready) {
  is_ready_ = is_ready;
}

int IAMRobolibStateInfo::get_watchdog_counter() {
  return watchdog_counter_;
}

void IAMRobolibStateInfo::reset_watchdog_counter() {
  watchdog_counter_ = 0;
}

void IAMRobolibStateInfo::increment_watchdog_counter() {
  watchdog_counter_++;
}

std::string IAMRobolibStateInfo::get_error_description() {
  std::string desc(error_description_,
                   error_description_ + sizeof(error_description_[0]) * error_description_len_);
  return desc;
}

void IAMRobolibStateInfo::set_error_description(std::string description){  
  std::memcpy(&error_description_, description.c_str(), description.size());
  error_description_len_  = description.size();
}

void IAMRobolibStateInfo::clear_error_description() {  
  set_error_description("");
}