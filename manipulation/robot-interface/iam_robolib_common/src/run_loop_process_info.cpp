//
// Created by mohit on 11/20/18.
//

#include <iam_robolib_common/run_loop_process_info.h>

void RunLoopProcessInfo::set_new_skill_available(bool new_skill_available) {
  new_skill_available_ = new_skill_available;
}

bool RunLoopProcessInfo::get_new_skill_available() {
  return new_skill_available_;
}

void RunLoopProcessInfo::set_new_skill_type(int type) {
  new_skill_type_ = type;
}

void RunLoopProcessInfo::set_new_meta_skill_type(int type) {
  new_meta_skill_type_ = type;
}

int RunLoopProcessInfo::get_new_skill_type() {
  return new_skill_type_;
}

int RunLoopProcessInfo::get_new_meta_skill_type() {
  return new_meta_skill_type_;
}

void RunLoopProcessInfo::set_is_running_skill(bool is_running_skill) {
  is_running_skill_ = is_running_skill;
}

bool RunLoopProcessInfo::get_is_running_skill() {
  return is_running_skill_;
}

void RunLoopProcessInfo::set_skill_preempted(bool skill_preempted) {
  skill_preempted_ = skill_preempted;
}

bool RunLoopProcessInfo::get_skill_preempted() {
  return skill_preempted_;
}

int RunLoopProcessInfo::get_current_shared_memory_index() {
  return current_memory_region_;
}

int RunLoopProcessInfo::get_current_free_shared_memory_index() {
  int current_free_memory_idx = 1 - current_memory_region_;
  return current_free_memory_idx;
}

void RunLoopProcessInfo::update_shared_memory_region() {
  assert(current_memory_region_ == 0 or current_memory_region_ == 1);
  current_memory_region_ = (current_memory_region_ + 1) % 2;
}

int RunLoopProcessInfo::get_current_shared_sensor_index() {
  return current_sensor_region_;
}

int RunLoopProcessInfo::get_current_free_shared_sensor_index() {
  int current_free_sensor_idx = 1 - current_sensor_region_;
  return current_free_sensor_idx;
}

void RunLoopProcessInfo::update_shared_sensor_region() {
  assert(current_sensor_region_ == 0 or current_sensor_region_ == 1);
  current_sensor_region_ = (current_sensor_region_ + 1) % 2;
}

int RunLoopProcessInfo::get_current_shared_feedback_index() {
  return current_feedback_region_;
}

int RunLoopProcessInfo::get_current_free_shared_feedback_index() {
  int current_free_feedback_idx = 1 - current_feedback_region_;
  return current_free_feedback_idx;
}

void RunLoopProcessInfo::update_shared_feedback_region() {
  assert(current_feedback_region_ == 0 or current_feedback_region_ == 1);
  current_feedback_region_ = (current_feedback_region_ + 1) % 2;
}

bool RunLoopProcessInfo::can_run_new_skill() {
  return is_running_skill_ == false;
}

int RunLoopProcessInfo::get_current_skill_id() {
  return current_skill_id_;
}

int RunLoopProcessInfo::get_current_meta_skill_id() {
  return current_meta_skill_id_;
}

void RunLoopProcessInfo::set_current_skill_id(int current_skill_id) {
  // Make sure we are updating to the latest available skill.
  assert(current_skill_id == new_skill_id_);
  current_skill_id_ = current_skill_id;
}

void RunLoopProcessInfo::set_current_meta_skill_id(int current_skill_id) {
  // Make sure we are updating to the latest available skill.
  assert(current_skill_id == new_meta_skill_id_);
  current_skill_id_ = current_skill_id;
}

int RunLoopProcessInfo::get_new_skill_id() {
  return new_skill_id_;
}

int RunLoopProcessInfo::get_new_meta_skill_id() {
  return new_meta_skill_id_;
}

void RunLoopProcessInfo::set_new_skill_id(int new_skill_id) {
  // Make sure we are getting the new skill
  assert(new_skill_id == current_skill_id_ + 1);
  new_skill_id_ = new_skill_id;
}

void RunLoopProcessInfo::set_new_meta_skill_id(int new_meta_skill_id) {
  new_meta_skill_id_ = new_meta_skill_id;
}

int RunLoopProcessInfo::get_done_skill_id() {
  return done_skill_id_;
}

void RunLoopProcessInfo::set_done_skill_id(int done_skill_id) {
  // Make sure we are getting the done skill
  assert(done_skill_id == current_skill_id_);
  done_skill_id_ = done_skill_id;
}

int RunLoopProcessInfo::get_result_skill_id() {
  return result_skill_id_;
}

void RunLoopProcessInfo::set_result_skill_id(int result_skill_id) {
  // Make sure we are getting the result skill
  assert(result_skill_id == done_skill_id_ ||
         result_skill_id == (done_skill_id_ - 1));
  result_skill_id_ = result_skill_id;
}
