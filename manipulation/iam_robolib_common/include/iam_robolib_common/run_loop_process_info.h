#pragma once

#include <cassert>
#include <string>

class RunLoopProcessInfo {
 public:
  RunLoopProcessInfo(int memory_region_idx): current_memory_region_(memory_region_idx) {};

  void set_new_skill_available(bool new_skill_available);

  bool get_new_skill_available();

  void set_new_skill_type(int type);

  void set_new_meta_skill_type(int type);

  int get_new_skill_type();

  int get_new_meta_skill_type();

  void set_is_running_skill(bool is_running_skill);

  bool get_is_running_skill();

  void set_skill_preempted(bool skill_preempted);

  bool get_skill_preempted();

/**
 * Memory index being used by the run loop.
 */
  int get_current_shared_memory_index();

/**
 * Memory index being used by the actionlib.
 */
  int get_current_free_shared_memory_index();

/**
 * Update shared memory region.
 */
  void update_shared_memory_region();

/**
 * Sensor index being used by the run loop.
 */
  int get_current_shared_sensor_index();

/**
 * Sensor index being used by the actionlib.
 */
  int get_current_free_shared_sensor_index();

/**
 * Update shared sensor region.
 */
  void update_shared_sensor_region();

/**
 * Feedback index being used by the run loop.
 */
  int get_current_shared_feedback_index();

/**
 * Feedback index being used by the actionlib.
 */
  int get_current_free_shared_feedback_index();

/**
 * Update shared feedback region.
 */
  void update_shared_feedback_region();

  bool can_run_new_skill();

/**
 * Return the id for the current skill.
 */
  int get_current_skill_id();

  /**
   * Return the id for the current meta skill.
   */
  int get_current_meta_skill_id();

/**
 * Set current skill id being executed.
 */
  void set_current_skill_id(int new_skill_id);

 /**
  * Set current skill id being executed.
  */
  void set_current_meta_skill_id(int new_skill_id);

/**
 * Return the id for the new skill.
 */
  int get_new_skill_id();

 /**
  * Return the id for the new skill.
  */
  int get_new_meta_skill_id();

/**
 * Set new skill id. Written from actionlib.
 */
  void set_new_skill_id(int new_skill_id);

 /**
  * Set new skill id. Written from actionlib.
  */
  void set_new_meta_skill_id(int new_meta_skill_id);

/**
 * Return the id for the done skill.
 */
  int get_done_skill_id();

/**
 * Set done skill id.
 */
  void set_done_skill_id(int done_skill_id);

/**
 * Return the id for the result skill.
 */
  int get_result_skill_id();

/**
 * Set result skill id.
 */
  void set_result_skill_id(int result_skill_id);

 private:
  bool new_skill_available_{false};
  int new_skill_type_{0};
  int new_meta_skill_type_{0};
  bool is_running_skill_{false};
  bool skill_preempted_{false};

  int current_memory_region_{1};
  int current_sensor_region_{1};
  int current_feedback_region_{1};
  int current_skill_id_{-1};
  int new_skill_id_{-1};
  int done_skill_id_{-1};
  int result_skill_id_{-1};

  int current_meta_skill_id_{-1};
  int new_meta_skill_id_{-1};
};

