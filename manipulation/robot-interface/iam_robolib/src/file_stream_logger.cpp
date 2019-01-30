//
// Created by mohit on 12/5/18.
//

#include "iam_robolib/file_stream_logger.h"

#include <iostream>

bool FileStreamLogger::writeData(std::vector<double> control_time,
               std::vector<std::array<double, 16>>& pose_desired,
               std::vector<std::array<double, 16>>& robot_state,
               std::vector<std::array<double, 7>>& tau_j,
               std::vector<std::array<double, 7>>& d_tau_j,
               std::vector<std::array<double, 7>>& q,
               std::vector<std::array<double, 7>>& q_d,
               std::vector<std::array<double, 7>>& dq) {
    if (!open_file_stream_.is_open()) {
        open_file_stream_ = std::ofstream(filename_, std::ofstream::out | std::ofstream::app);
    }

    bool all_sizes_equal = true;
    size_t control_time_size = control_time.size();
    if (write_pose_desired_ && control_time_size != pose_desired.size()) {
        all_sizes_equal = false;
        std::cout << "Control time and pose desired size do not match\n";
    } else if (control_time_size != robot_state.size()) {
        all_sizes_equal = false;
        std::cout << "Control time and robot state size do not match\n";
    } else if (control_time_size != tau_j.size()) {
        all_sizes_equal = false;
        std::cout << "Control time and tau_j size do not match\n";
    } else if (control_time_size != d_tau_j.size()) {
        all_sizes_equal = false;
        std::cout << "Control time and d_tau_j size do not match\n";
    } else if (control_time_size != q.size()) {
        all_sizes_equal = false;
        std::cout << "Control time and q size do not match\n";
    } else if (control_time_size != dq.size()) {
        all_sizes_equal = false;
        std::cout << "Control time and dq size do not match\n";
    }

    if (!all_sizes_equal) {
        std::cout << "Save vectors do not have same size. Will not save!!!" << std::endl;
        // Can throw error
        return false;
    }

    for (int i = 0; i < control_time_size; i++) {
        open_file_stream_ << control_time[i] << ",";
        if (write_pose_desired_) {
            std::array<double, 16> &pose = pose_desired[i];
            for (const auto &e : pose) {
                open_file_stream_ << e << ",";
            }
        }
        std::array<double, 16>& pose = robot_state[i];
        for (const auto &e : pose) {
            open_file_stream_ << e << ",";
        }
        //  Log tau_j and d_tau_j
        std::array<double, 7> &state = tau_j[i];
        for (const auto &e : state) {
            open_file_stream_ << e << ",";
        }
        state = d_tau_j[i];
        for (const auto &e : state) {
            open_file_stream_ << e << ",";
        }

        // Log q and dq
        state = q[i];
        for (const auto &e : state) {
            open_file_stream_ << e << ",";
        }
        state = q_d[i];
        for (const auto &e : state) {
            open_file_stream_ << e << ",";
        }

        state = dq[i];
        for (const auto &e : state) {
            open_file_stream_ << e << ",";
        }
        open_file_stream_ << "\n";
    }
    open_file_stream_.close();
    return true;
}

bool FileStreamLogger::writeStringData(std::vector<std::string> data) {
    if (!open_file_stream_.is_open()) {
        open_file_stream_ = std::ofstream(filename_, std::ofstream::out | std::ofstream::app);
    }
    size_t data_size = data.size();
    for (int i = 0; i < data_size ; i++) {
        open_file_stream_ << "info: " << data[i] << "\n";
    }
}
