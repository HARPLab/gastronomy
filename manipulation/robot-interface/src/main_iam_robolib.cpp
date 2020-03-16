#include <iostream>
#include <mutex>
#include <boost/program_options.hpp>

#include <iam_robolib_common/definitions.h>

#include <iam_robolib/run_loop.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {

  try {
    int robot_type_int;
    int stop_robolib_on_error;
    int reset_skill_numbering_on_error;
    int use_new_filestream_on_error;
    std::string logdir;
    RobotType robot_type;
    std::string robot_ip;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("robot_type", po::value<int>(&robot_type_int)->default_value(0),
            "robot type: 0 for Franka and 1 for UR5e")
      ("robot_ip,ip_addr,ip", po::value<std::string>(&robot_ip)->default_value("172.16.0.2"),
            "robot's ip address")
      ("stop_on_error", po::value<int>(&stop_robolib_on_error)->default_value(0),
            "Stop robo-lib on error, i.e. any exception thrown by libfranka.")
      ("reset_skill_numbering_on_error", po::value<int>(&reset_skill_numbering_on_error)->default_value(0),
            "Reset skill numbering on error, i.e. any exception thrown by libfranka.")
      ("use_new_filestream_on_error", po::value<int>(&use_new_filestream_on_error)->default_value(0),
            "Use a new filestream on error, i.e. any exception thrown by libfranka.")
      ("logdir", po::value<std::string>(&logdir)->default_value("logs"), "directory to save robot_state_data")
    ;

    po::positional_options_description p;
    p.add("robot_ip", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << "Usage: options_description [options]\n";
        cout << desc;
        return 0;
    }

    robot_type = (RobotType) robot_type_int;

    std::cout << "IAM Robolib\n";
    std::mutex m;
    std::mutex robot_loop_data_mutex;
    run_loop rl = run_loop(std::ref(m), std::ref(robot_loop_data_mutex), robot_type, robot_ip,
        stop_robolib_on_error, reset_skill_numbering_on_error, use_new_filestream_on_error, logdir);
    std::cout << "Will start run loop.\n";
    
    switch(robot_type)
    {
      case RobotType::FRANKA:
        rl.start();
        std::cout << "Did start run loop.\n";
        std::cout << "Will run..\n";
        rl.run_on_franka();
        break;
      case RobotType::UR5E:
        rl.start_ur5e();
        std::cout << "Did start run loop.\n";
        std::cout << "Will run..\n";
        rl.run_on_ur5e();
        break;
    }
    
  }
  catch(std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }
  
  return 0;
}
