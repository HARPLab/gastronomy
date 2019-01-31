#include "iam_robolib/run_loop.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <pthread.h>

#include <cerrno>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <cassert>

#include <franka/exception.h>

#include "iam_robolib/duration.h"
#include "iam_robolib/skills/base_meta_skill.h"
#include "iam_robolib/skills/base_skill.h"
#include "iam_robolib/robot_state_data.h"
#include "iam_robolib/file_stream_logger.h"
#include "iam_robolib/skills/gripper_open_skill.h"
#include "iam_robolib/skills/joint_pose_skill.h"
#include "iam_robolib/skills/joint_pose_continuous_skill.h"
#include "iam_robolib/skills/save_trajectory_skill.h"
#include "iam_robolib/skills/force_torque_skill.h"

std::atomic<bool> run_loop::running_skills_{false};

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args )
{
  size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
  std::unique_ptr<char[]> buf( new char[ size ] );
  snprintf( buf.get(), size, format.c_str(), args ... );
  return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void setCurrentThreadToRealtime(bool throw_on_error) {
  // Change prints to exceptions.
  const int thread_priority = sched_get_priority_max(SCHED_FIFO);
  if (thread_priority == -1) {
    std::cout << std::string("libfranka: unable to get maximum possible thread priority: ") +
        std::strerror(errno);
  }
  sched_param thread_param{};
  thread_param.sched_priority = thread_priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &thread_param) != 0) {
    if (throw_on_error) {
      std::cout << std::string("libfranka: unable to set realtime scheduling: ") +
          std::strerror(errno);
    }
  }
}

bool run_loop::init() {
  // TODO(Mohit): Initialize memory and stuff.
  bool throw_on_error;
  setCurrentThreadToRealtime(throw_on_error);
  shared_memory_handler_ = new RunLoopSharedMemoryHandler();
}

void run_loop::start() {
  init();
  // Start processing, might want to do some pre-processing 
  std::cout << "start run loop.\n";
  shared_memory_handler_->start();
}

void run_loop::start_ur5e() {
  init();
  // Start processing, might want to do some pre-processing 
  std::cout << "start run loop.\n";
  shared_memory_handler_->start();

  dynamic_cast<UR5eRobot* >(robot_)->rt_pl_->run();
  dynamic_cast<UR5eRobot* >(robot_)->rt_transmit_stream_.connect();
}

void run_loop::stop() {
  // Maybe call this after exceptions or SIGINT or any Interrupt.
  // Stop the interface gracefully.
}

bool run_loop::should_start_new_skill(BaseSkill* old_skill, BaseSkill* new_skill) {
  // No new skill to start.
  if (new_skill == nullptr) {
    return false;
  }
  // Old  skill was null, new skill is not null. should start it.
  if (old_skill == nullptr) {
    return true;
  }
  // If new skill is different than old skill, we should start it.
  if (new_skill->get_skill_id() != old_skill->get_skill_id()) {
    return true;
  }

  return false;
}

void run_loop::start_new_skill(BaseSkill* new_skill) {
  // Generate things that are required here.
  RunLoopProcessInfo* run_loop_info = shared_memory_handler_->getRunLoopProcessInfo();
  int memory_index = run_loop_info->get_current_shared_memory_index();
  std::cout << string_format("Create skill from memory index: %d\n", memory_index);

  SharedBuffer traj_buffer = shared_memory_handler_->getTrajectoryGeneratorBuffer(memory_index);
  TrajectoryGenerator *traj_generator = traj_gen_factory_.getTrajectoryGeneratorForSkill(
      traj_buffer);
  std::cout << "Did get traj generator\n";

  SharedBuffer feedback_controller_buffer = shared_memory_handler_->getFeedbackControllerBuffer(
      memory_index);
  FeedbackController *feedback_controller =
      feedback_controller_factory_.getFeedbackControllerForSkill(feedback_controller_buffer);

  SharedBuffer termination_handler_buffer = shared_memory_handler_->getTerminationParametersBuffer(
      memory_index);
  TerminationHandler* termination_handler =
      termination_handler_factory_.getTerminationHandlerForSkill(termination_handler_buffer, run_loop_info);
  std::cout << "Did get TerminationHandler\n";

  // Start skill, does any pre-processing if required.
  
  new_skill->start_skill(robot_, traj_generator, feedback_controller, termination_handler);
}

void run_loop::finish_current_skill(BaseSkill* skill) {
  SkillStatus status = skill->get_current_skill_status();

  if (skill->should_terminate()) {
    skill->set_skill_status(SkillStatus::FINISHED);

    // Write results to memory
    int memory_index = skill->get_skill_id() % 2;

    std::cout << "Writing to execution result buffer number: " << memory_index << std::endl;

    SharedBuffer buffer = shared_memory_handler_->getExecutionResultBuffer(memory_index);
    skill->write_result_to_shared_memory(buffer, robot_);
  }

  if (status == SkillStatus::FINISHED) {
    process_info_requires_update_ = true;
  }
  // TODO(Mohit): Do any other-preprocessing if required
}

void run_loop::update_process_info() {
  BaseSkill* skill = skill_manager_.get_current_skill();
  int current_skill_id = -1;
  if (skill != nullptr) {
    current_skill_id = skill->get_skill_id();
  }
  bool is_executing_skill = skill_manager_.is_currently_executing_skill();

  // Grab the lock and update process info.
  {
    RunLoopProcessInfo* run_loop_info = shared_memory_handler_->getRunLoopProcessInfo();
    boost::interprocess::scoped_lock<
            boost::interprocess::interprocess_mutex> lock(
                *(shared_memory_handler_->getRunLoopProcessInfoMutex()),
                boost::interprocess::defer_lock);
    try {
      if (lock.try_lock()) {
        run_loop_info->set_is_running_skill(is_executing_skill);

        // We have a skill that we have finished. Make sure we update this in RunLoopProcessInfo.
        if (skill != nullptr && !is_executing_skill) {
          if (run_loop_info->get_done_skill_id() > current_skill_id) {
            // Make sure get done skill id is not ahead of us.
            std::cout << string_format("INVALID: RunLoopProcInfo has done skill id %d "
                                                " greater than current skill id %d\n",
                                                run_loop_info->get_done_skill_id(),
                                                current_skill_id);
          } else if (run_loop_info->get_result_skill_id() + 2 <= current_skill_id) {
            // Make sure that ActionLib has read the skill results before we overwrite them.
            std::cout << string_format("ActionLib server has not read previous result %d. "
                              "Cannot write new result %d\n",
                              run_loop_info->get_result_skill_id(),
                              current_skill_id);
          } else if (run_loop_info->get_done_skill_id() != current_skill_id - 1) {
            // Make sure we are only updating skill sequentially.
            std::cout << string_format("RunLoopProcInfo done skill id: %d current skill id: %d\n",
                    run_loop_info->get_done_skill_id(), current_skill_id);
          } else {
            run_loop_info->set_done_skill_id(current_skill_id);
            std::cout << string_format("Did set done_skill_id %d\n", current_skill_id);
          }
        }
        process_info_requires_update_ = false;

        // Check if new skill is available only if no current skill is being
        // currently executed.
        if (!is_executing_skill && run_loop_info->get_new_skill_available()) {

          std::cout << "Did get new skill";
          // Create new task Skill
          int new_skill_id = run_loop_info->get_new_skill_id();
          int new_skill_type = run_loop_info->get_new_skill_type();
          int new_meta_skill_id = run_loop_info->get_new_meta_skill_id();
          int new_meta_skill_type = run_loop_info->get_new_meta_skill_type();
          std::string new_skill_description = run_loop_info->get_new_skill_description();
          std::cout << string_format("Did find new skill id: %d, type: %d meta skill: %d, type: %d\n",
              new_skill_id, new_skill_type, new_meta_skill_id, new_meta_skill_type);

          // Add new skill
          run_loop_info->set_current_skill_id(new_skill_id);
          BaseSkill *new_skill;
          if (new_skill_type == 0) {
            new_skill = new SkillInfo(new_skill_id, new_meta_skill_id, new_skill_description);
          } else if (new_skill_type == 1) {
            new_skill = new GripperOpenSkill(new_skill_id, new_meta_skill_id, new_skill_description);
          } else if (new_skill_type == 2) {
            new_skill = new JointPoseSkill(new_skill_id, new_meta_skill_id, new_skill_description);
          } else if (new_skill_type == 3) {
            new_skill = new SaveTrajectorySkill(new_skill_id, new_meta_skill_id, new_skill_description);
          } else if (new_skill_type == 4) {
            new_skill = new ForceTorqueSkill(new_skill_id, new_meta_skill_id, new_skill_description);
          } else {
              std::cout << "Incorrect skill type: " << new_skill_type << "\n";
              assert(false);
          }
          skill_manager_.add_skill(new_skill);

          // Get Meta-skill
          // BaseMetaSkill* new_meta_skill = skill_manager_.get_meta_skill_with_id(new_meta_skill_id);
           BaseMetaSkill* new_meta_skill = nullptr;
          if (new_meta_skill == nullptr) {
            if (new_meta_skill_type == 0) {
              new_meta_skill = new BaseMetaSkill(new_meta_skill_id);
            } else if (new_meta_skill_type == 1) {
              new_meta_skill = new JointPoseContinuousSkill(new_meta_skill_id);
            } else {
                std::cout << "Incorrect meta skill type: " << new_skill_type << "\n";
                assert(false);
            }
            skill_manager_.add_meta_skill(new_meta_skill);
          }

          // Update the shared memory region. This means that the actionlib service will now write
          // to the other memory region, i.e. not the current memory region.
          // TODO(Mohit): We should lock the other memory so that ActionLibServer cannot modify it?
          run_loop_info->update_shared_memory_region();
          run_loop_info->set_new_skill_available(false);
        } else {          
          std::cout << "Did not get new skill\n";
        }
      }
    } catch (boost::interprocess::lock_exception) {
      // TODO(Mohit): Do something better here.
      std:: cout << "Cannot acquire lock for run loop info";
    }
  }
}

void run_loop::run() {
  // Wait for sometime to let the client add data to the buffer
  std::this_thread::sleep_for(std::chrono::seconds(10));

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  auto milli = std::chrono::milliseconds(1);

  RunLoopProcessInfo* run_loop_info = shared_memory_handler_->getRunLoopProcessInfo();
  while (1) {
    start = std::chrono::high_resolution_clock::now();

    // Execute the current skill (traj_generator, FBC are here)
    BaseSkill *skill = skill_manager_.get_current_skill();

    // NOTE: We keep on running the last skill even if it is finished!!
    if (skill != 0) {
      // Execute skill.
      skill->execute_skill();

      int memory_index = run_loop_info->get_current_shared_memory_index();
      SharedBuffer buffer = shared_memory_handler_->getFeedbackResultBuffer(memory_index);
      skill->write_feedback_to_shared_memory(buffer);

      // Finish skill if possible.
      finish_current_skill(skill);
    }

    // Complete old skills and acquire new skills
    update_process_info();

    // Start new skill, if possible
    BaseSkill *new_skill = skill_manager_.get_current_skill();
    if (should_start_new_skill(skill, new_skill)) {
      start_new_skill(new_skill);
    }

    // Sleep to maintain 1Khz frequency, not sure if this is required or not.
    auto finish = std::chrono::high_resolution_clock::now();
    // Wait for start + milli - finish
    auto elapsed = start + milli - finish;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void run_loop::setup_save_robot_state_thread() {
  int print_rate = 100;   // The below thread will print at 10 FPS.
  print_thread_ = std::thread([&, print_rate]() {
      // Sleep to achieve the desired print rate.
      std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
      while (running_skills_) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0))));
        // Try to lock data to avoid read write collisions.
        try {
          switch(robot_->robot_type_)
          {
            case RobotType::FRANKA: {
                franka::RobotState robot_state = dynamic_cast<FrankaRobot* >(robot_)->getRobotState();
                // franka::GripperState gripper_state = gripper_.readOnce();
                // TODO(jacky): is this duration still needed?
                double duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                robot_state_data_->log_pose_desired(robot_state.O_T_EE_d); // Fictitious call to log pose desired so pose_desired buffer length matches during non-skill execution
                robot_state_data_->log_robot_state(robot_state, duration / 1000.0);
                // robot_state_data_->log_gripper_state(gripper_state);
                // std::cout << duration / 1000.0 << "\n";
              }
              break;
            case RobotType::UR5E:
              break;
          }
          
        } catch (const franka::InvalidOperationException& ex) {
          // Some other control thread is running let's wait and try again.
          std::cerr << "Cannot read robot state for logging. Will continue. " << ex.what() << std::endl;
        }
      }
  });
}

void run_loop::setup_current_robot_state_io_thread() {
  int io_rate = 100;
  current_robot_state_io_thread_ = std::thread([&, io_rate]() {
      std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

      while (running_skills_) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>((1.0 / io_rate * 1000.0))));

          if (robot_state_data_ == nullptr || robot_state_data_->log_robot_state_0_.size() == 0) continue;

          // Try to lock data to avoid read write collisions.
          
          if (robot_state_data_->use_buffer_0) {
            if (robot_state_data_->buffer_0_mutex_.try_lock()) {
              if (shared_memory_handler_->getCurrentRobotStateBufferMutex()->try_lock()) {
                  float* current_robot_state_data_buffer = shared_memory_handler_->getCurrentRobotStateBuffer();
                  size_t buffer_idx = 0;
                  double double_val = 0;
                  std::array<double, 16> double_array_16;
                  std::array<double, 7> double_array_7;

                  double_array_16 = robot_state_data_->log_pose_desired_0_.back();
                  for (size_t i = 0; i < double_array_16.size(); i++) {
                    double_val = double_array_16[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_16 = robot_state_data_->log_robot_state_0_.back();
                  for (size_t i = 0; i < double_array_16.size(); i++) {
                    double_val = double_array_16[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_tau_j_0_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_d_tau_j_0_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_q_0_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_q_d_0_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_dq_0_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_val = robot_state_data_->log_control_time_0_.back();
                  current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);

                  if (robot_state_data_->log_gripper_width_0_.size() > 0) {
                    double_val = robot_state_data_->log_gripper_width_0_.back();
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);

                    double_val = robot_state_data_->log_gripper_is_grasped_0_.back() ? 1 : 0;
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  } else {
                    current_robot_state_data_buffer[buffer_idx++] = -1.f;
                    current_robot_state_data_buffer[buffer_idx++] = 0.f;
                  }

                  shared_memory_handler_->getCurrentRobotStateBufferMutex()->unlock();
              }
              robot_state_data_->buffer_0_mutex_.unlock();
            }
          }
          else {
            if (robot_state_data_->buffer_1_mutex_.try_lock()) {
              if (shared_memory_handler_->getCurrentRobotStateBufferMutex()->try_lock()) {
                  float* current_robot_state_data_buffer = shared_memory_handler_->getCurrentRobotStateBuffer();
                  size_t buffer_idx = 0;
                  double double_val = 0;
                  std::array<double, 16> double_array_16;
                  std::array<double, 7> double_array_7;

                  double_array_16 = robot_state_data_->log_pose_desired_1_.back();
                  for (size_t i = 0; i < double_array_16.size(); i++) {
                    double_val = double_array_16[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_16 = robot_state_data_->log_robot_state_1_.back();
                  for (size_t i = 0; i < double_array_16.size(); i++) {
                    double_val = double_array_16[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_tau_j_1_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_d_tau_j_1_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_q_1_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_q_d_1_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_array_7 = robot_state_data_->log_dq_1_.back();
                  for (size_t i = 0; i < double_array_7.size(); i++) {
                    double_val = double_array_7[i];
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  }

                  double_val = robot_state_data_->log_control_time_1_.back();
                  current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);

                  if (robot_state_data_->log_gripper_width_1_.size() > 0) {
                    double_val = robot_state_data_->log_gripper_width_1_.back();
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);

                    double_val = robot_state_data_->log_gripper_is_grasped_1_.back() ? 1 : 0;
                    current_robot_state_data_buffer[buffer_idx++] = static_cast<float> (double_val);
                  } else {
                    current_robot_state_data_buffer[buffer_idx++] = -1.f;
                    current_robot_state_data_buffer[buffer_idx++] = 0.f;
                  }

                  shared_memory_handler_->getCurrentRobotStateBufferMutex()->unlock();
              }
              robot_state_data_->buffer_1_mutex_.unlock();
            }
          }
      }
  });
}

void run_loop::setup_robot_default_behavior() {
  switch(robot_->robot_type_)
  {
    case RobotType::FRANKA: {
        // Set additional parameters always before the control loop, NEVER in the control loop!
        // Set collision behavior.
        /*robot_->robot_.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});*/
        dynamic_cast<FrankaRobot* >(robot_)->robot_.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{120.0, 120.0, 118.0, 118.0, 116.0, 114.0, 112.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{120.0, 120.0, 118.0, 118.0, 116.0, 114.0, 112.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{120.0, 120.0, 120.0, 125.0, 125.0, 125.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{120.0, 120.0, 120.0, 125.0, 125.0, 125.0}});

        dynamic_cast<FrankaRobot* >(robot_)->robot_.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
        dynamic_cast<FrankaRobot* >(robot_)->robot_.setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
      }
      break;
    case RobotType::UR5E:
      break;
  }
  
}

void run_loop::didFinishSkillInMetaSkill(BaseSkill* skill) {
  // Finish skill if possible.
  finish_current_skill(skill);
  // Complete old skills and acquire new skills
  update_process_info();
}

void run_loop::setup_data_loggers() {
  FileStreamLogger *robot_logger = new FileStreamLogger("./robot_state_data.txt");
  robot_state_data_->setFileStreamLogger(robot_logger);
  robot_state_data_->startFileLoggerThread();
}

void run_loop::log_skill_info(BaseSkill* skill) {
  std::string log_desc = string_format("Will execute skill: %d, meta_skill: %d, ",
                                       skill->get_skill_id(), skill->get_meta_skill_id());
  log_desc += ("desc: " + skill->get_description());
  robot_state_data_->log_skill_info(log_desc);
  std::cout << log_desc << "\n" << std::endl;
};

void run_loop::run_on_franka() {
  // Wait for sometime to let the client add data to the buffer
  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  auto milli = std::chrono::milliseconds(1);

  setup_robot_default_behavior();

  try {
    running_skills_ = true;
    setup_data_loggers();
    setup_current_robot_state_io_thread();
    // TODO(Mohit): This causes a weird race condition between reading the robot state and 
    // running the control loop. It prevents iam_robolib from running when robot is in guide mode.
    // setup_save_robot_state_thread();

    while (1) {
      start = std::chrono::high_resolution_clock::now();

      // Execute the current skill (traj_generator, FBC are here)
      BaseSkill* skill = skill_manager_.get_current_skill();
      BaseMetaSkill *meta_skill = skill_manager_.get_current_meta_skill();

      // NOTE: We keep on running the last skill even if it is finished!!
      if (skill != nullptr && meta_skill != nullptr) {
        if (!meta_skill->isComposableSkill() && !skill->get_termination_handler()->done_) {
          // Execute skill.
          log_skill_info(skill);
          meta_skill->execute_skill_on_franka(this, dynamic_cast<FrankaRobot* >(robot_), robot_state_data_);
        } else if (meta_skill->isComposableSkill()) {
          log_skill_info(skill);
          meta_skill->execute_skill_on_franka(this, dynamic_cast<FrankaRobot* >(robot_), robot_state_data_);
        } else {
          finish_current_skill(skill);
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      // Complete old skills and acquire new skills
      update_process_info();

      // Start new skill, if possible
      BaseSkill* new_skill = skill_manager_.get_current_skill();
      if (should_start_new_skill(skill, new_skill)) {
        std::cout << "Will start skill\n";
        start_new_skill(new_skill);
      }

      // Sleep to maintain 1Khz frequency, not sure if this is required or not.
      auto finish = std::chrono::high_resolution_clock::now();
      // Wait for start + milli - finish
      auto elapsed = start + milli - finish;
      // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } catch (const franka::Exception& ex) {
    std::cout << "Franka exception occurred during control loop. Will exit." << std::endl;
    std::cerr << ex.what() << std::endl;
    logger_.print_error_log();
    logger_.print_warning_log();
    logger_.print_info_log();
  }

  if (print_thread_.joinable()) {
    print_thread_.join();
  }
  if (current_robot_state_io_thread_.joinable()) {
    current_robot_state_io_thread_.join();
  }
  if (robot_state_data_->file_logger_thread_.joinable()) {
    robot_state_data_->file_logger_thread_.join();
  }
}

void run_loop::run_on_ur5e() {
  // Wait for sometime to let the client add data to the buffer
  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  auto milli = std::chrono::milliseconds(1);

  std::array<double, 6> pose = {0.13339,-0.49242,0.48877,0.0,3.136,0.0};
  //std::array<double, 6> joints = {1.57,-1.57,1.57,-1.57,-1.57,0};
  double tool_acceleration = 1.0;
  //double gain = 300.0;

  int i = 0;

  LOG_INFO("Starting main loop");

  while(i < 10000)
  {
    //joints[5] += 0.0003;
    pose[2] -= 0.00001;
    //rt_commander->servoj(joints, gain);

    dynamic_cast<UR5eRobot* >(robot_)->rt_commander_->movel(pose, tool_acceleration);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    i++;
    //std::cout << i << std::endl;
  }

  dynamic_cast<UR5eRobot* >(robot_)->rt_commander_->stopl();

  // setup_robot_default_behavior();

  /*try {
    running_skills_ = true;

    //setup_data_loggers();
    //setup_current_robot_state_io_thread();

    // TODO(Mohit): This causes a weird race condition between reading the robot state and 
    // running the control loop. It prevents iam_robolib from running when robot is in guide mode.
    // setup_save_robot_state_thread();

    LOG_INFO("Starting main loop");

    while (1) {
      start = std::chrono::high_resolution_clock::now();

      // Execute the current skill (traj_generator, FBC are here)
      BaseSkill* skill = skill_manager_.get_current_skill();
      BaseMetaSkill *meta_skill = skill_manager_.get_current_meta_skill();

      // NOTE: We keep on running the last skill even if it is finished!!
      if (skill != nullptr && meta_skill != nullptr) {
        if (!meta_skill->isComposableSkill() && !skill->get_termination_handler()->done_) {
          // Execute skill.
          log_skill_info(skill);
          //meta_skill->execute_skill_on_franka(this, dynamic_cast<FrankaRobot* >(robot_), robot_state_data_);
        } else if (meta_skill->isComposableSkill()) {
          log_skill_info(skill);
          //meta_skill->execute_skill_on_franka(this, dynamic_cast<FrankaRobot* >(robot_), robot_state_data_);
        } else {
          finish_current_skill(skill);
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      // Complete old skills and acquire new skills
      update_process_info();

      // Start new skill, if possible
      BaseSkill* new_skill = skill_manager_.get_current_skill();
      if (should_start_new_skill(skill, new_skill)) {
        std::cout << "Will start skill\n";
        start_new_skill(new_skill);
      }

      // Sleep to maintain 1Khz frequency, not sure if this is required or not.
      auto finish = std::chrono::high_resolution_clock::now();
      // Wait for start + milli - finish
      auto elapsed = start + milli - finish;
      // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } catch (const franka::Exception& ex) {
    std::cout << "Franka exception occurred during control loop. Will exit." << std::endl;
    std::cerr << ex.what() << std::endl;
    logger_.print_error_log();
    logger_.print_warning_log();
    logger_.print_info_log();
  }*/

  /*if (print_thread_.joinable()) {
    print_thread_.join();
  }
  if (current_robot_state_io_thread_.joinable()) {
    current_robot_state_io_thread_.join();
  }
  if (robot_state_data_->file_logger_thread_.joinable()) {
    robot_state_data_->file_logger_thread_.join();
  }*/

  LOG_INFO("Stopping, shutting down pipelines");

  dynamic_cast<UR5eRobot* >(robot_)->rt_transmit_stream_.disconnect();
  dynamic_cast<UR5eRobot* >(robot_)->rt_pl_->stop();
}

SkillInfoManager* run_loop::getSkillInfoManager() {
  return &skill_manager_;
}
