//
// Created by mohit on 11/20/18.
//

#include "run_loop_process_info.h"

#include <cassert>

int RunLoopProcessInfo::get_current_shared_memory_index() {
    return current_memory_region_;
}

std::string RunLoopProcessInfo::get_shared_memory_name_for_memory_idx(
    int memory_idx) {
  assert(memory_idx == 0 or memory_idx == 1);
  return "run_loop_shared_memory_" + memory_idx;
}

std::string RunLoopProcessInfo::get_current_shared_memory_name() {
  return get_shared_memory_name_for_memory_idx(current_memory_region_);
}

std::string RunLoopProcessInfo::get_shared_memory_for_actionlib() {
  int memory_idx = 1 - current_memory_region_;
  return get_shared_memory_name_for_memory_idx(memory_idx);
}

bool RunLoopProcessInfo::can_run_new_task() {
    return is_running_task_ == false;
}

int RunLoopProcessInfo::get_new_skill_id() {
  return new_skill_id_;
}

void RunLoopProcessInfo::update_current_skill(int new_skill_id) {
  // Make sure we are updating to the latest available skill.
  assert(new_skill_id == new_skill_id_);
  current_skill_id_ = new_skill_id;
}

void RunLoopProcessInfo::update_shared_memory_region() {
  assert(current_memory_region_ == 0 or current_memory_region_ == 1);
  current_memory_region_ = (current_memory_region_ + 1) % 2;
}

void RunLoopProcessInfo::update_new_skill(int new_skill_id) {
  // Make sure we are getting the new skill
  assert(new_skill_id > current_skill_id_);
  new_skill_id_ = new_skill_id;
}
