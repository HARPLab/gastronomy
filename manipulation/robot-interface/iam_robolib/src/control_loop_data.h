#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <vector>
#include <thread>

#include <franka/robot.h>

class FileStreamLogger;

class ControlLoopData {
 public:
  std::atomic<bool> use_buffer_0;
  std::mutex buffer_0_mutex_;
  std::mutex buffer_1_mutex_;

  ControlLoopData(std::mutex &m): mutex_(m) {};

  std::mutex& mutex_;
  bool has_data_=false;

  double time_=0;
  int counter_=0;

  std::vector<std::array<double, 16>> log_pose_desired_0_{};
  std::vector<std::array<double, 16>> log_robot_state_0_{};
  std::vector<std::array<double, 7>> log_tau_j_0_{};
  std::vector<std::array<double, 7>> log_d_tau_j_0_{};
  std::vector<std::array<double, 7>> log_q_0_{};
  std::vector<std::array<double, 7>> log_q_d_0_{};
  std::vector<std::array<double, 7>> log_dq_0_{};
  std::vector<double> log_control_time_0_{};

  std::vector<std::array<double, 16>> log_pose_desired_1_{};
  std::vector<std::array<double, 16>> log_robot_state_1_{};
  std::vector<std::array<double, 7>> log_tau_j_1_{};
  std::vector<std::array<double, 7>> log_d_tau_j_1_{};
  std::vector<std::array<double, 7>> log_q_1_{};
  std::vector<std::array<double, 7>> log_q_d_1_{};
  std::vector<std::array<double, 7>> log_dq_1_{};
  std::vector<double> log_control_time_1_{};

  //These act as global buffers
  std::vector<std::array<double, 16>> log_pose_desired_g_{};
  std::vector<std::array<double, 16>> log_robot_state_g_{};
  std::vector<std::array<double, 7>> log_tau_j_g_={};
  std::vector<std::array<double, 7>> log_dq_g_{};
  std::vector<double> log_control_time_g_{};

  // Utils for printing
  const int print_rate_=10;

  const int log_rate_=10;

  std::thread file_logger_thread_;

  /**
   * Set filestream logger to save data.
   * @param logger
   */
  void setFileStreamLogger(FileStreamLogger *logger);

  /**
   * Start logging to some global buffer or file.
   */
  void startFileLoggerThread();

  /**
   * Force write everything in current buffer to global buffer.
   * Use this to make sure we do not lose any data when we crash.
   */
  void writeCurrentBufferData();

  /**
   * Print Data beginning from end.
   * @param print_count
   */
  void printGlobalData(int print_count);

  /**
   * Print measured jerks from dq (joint velocities) data.
   * @param data Joint velocities
   * @param print_last  number of last values to print.
   */
  void printMeasuredJointJerks(std::vector<std::array<double, 7>> data, int print_last);

  /*
   * Helper methods for logging.
   */
  void log_pose_desired(std::array<double, 16> pose_desired_);
  void log_robot_state(franka::RobotState robot_state, double time);

 private:
  FileStreamLogger *file_logger_ = nullptr;

  void writeBufferData_0();

  void writeBufferData_1();
 };