#ifndef IAM_ROBOLIB_FILE_STREAM_LOGGER_H_
#define IAM_ROBOLIB_FILE_STREAM_LOGGER_H_

#include <array>
#include <fstream>
#include <vector>

class FileStreamLogger {
 public:
  FileStreamLogger(const std::string& filename): filename_(filename),
                                                 open_file_stream_(filename, std::ofstream::out | std::ofstream::app) {};

  bool write_pose_desired_=true;

  bool writeData(std::vector<double> control_time,
                 std::vector<std::array<double, 16>>& pose_desired,
                 std::vector<std::array<double, 16>>& robot_state,
                 std::vector<std::array<double, 7>>& tau_j,
                 std::vector<std::array<double, 7>>& d_tau_j,
                 std::vector<std::array<double, 7>>& q,
                 std::vector<std::array<double, 7>>& q_d,
                 std::vector<std::array<double, 7>>& dq);

  /**
   * Write string data to logger. String data is prefixed by "info:" to allow us to easily find it in the logs.
   * @param data String data to log does not require a "\n" in the end.
   * @return True if we did write the data successfully else false.
   */
  bool writeStringData(std::vector<std::string> data);

 private:
  std::ofstream open_file_stream_;
  std::string filename_;
};

#endif  // IAM_ROBOLIB_FILE_STREAM_LOGGER_H_