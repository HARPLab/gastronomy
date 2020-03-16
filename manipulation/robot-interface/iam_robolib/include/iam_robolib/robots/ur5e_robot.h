#ifndef IAM_ROBOLIB_ROBOTS_UR5E_ROBOT_H_
#define IAM_ROBOLIB_ROBOTS_UR5E_ROBOT_H_

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <iostream>

#include "ur_modern_driver/log.h"
#include "ur_modern_driver/pipeline.h"
#include "ur_modern_driver/ur/commander.h"
#include "ur_modern_driver/ur/factory.h"
#include "ur_modern_driver/ur/messages.h"
#include "ur_modern_driver/ur/parser.h"
#include "ur_modern_driver/ur/producer.h"
#include "ur_modern_driver/ur/rt_state.h"
#include "ur_modern_driver/ur/rt_consumer.h"
#include "ur_modern_driver/ur/state.h"

#include "iam_robolib/robots/robot.h"

class IgnorePipelineStoppedNotifier : public INotifier
{
public:
    void started(std::string name){
        LOG_INFO("Starting pipeline %s", name.c_str());
    }
    void stopped(std::string name){
        LOG_INFO("Stopping pipeline %s", name.c_str());
    }
};

class ShutdownOnPipelineStoppedNotifier : public INotifier
{
public:
    void started(std::string name){
        LOG_INFO("Starting pipeline %s", name.c_str());
    }
    void stopped(std::string name){
        LOG_INFO("Shutting down on stopped pipeline %s", name.c_str());
        exit(1);
    }
};

class UR5eRobot : public Robot
{
 public:
  UR5eRobot(std::string &robot_ip, RobotType robot_type) : 
                            Robot(robot_ip, robot_type),
                            rt_transmit_stream_(robot_ip, UR_RT_TRANSMIT_PORT_),
                            factory_(robot_ip),
                            rt_receive_stream_(robot_ip, UR_RT_RECEIVE_PORT_)
  {
    std::unique_ptr<URParser<RTPacket>> rt_parser = factory_.getRTParser();
    URProducer<RTPacket> rt_prod(rt_receive_stream_, *rt_parser);
    RTConsumer rt_consumer(false);
    
    INotifier *notifier(nullptr);
    if (true)
    {
      LOG_INFO("Notifier: Pipeline disconnect will shutdown the node");
      notifier = new ShutdownOnPipelineStoppedNotifier();
    }
    else
    {
      LOG_INFO("Notifier: Pipeline disconnect will be ignored.");
      notifier = new IgnorePipelineStoppedNotifier();
    }

    vector<IConsumer<RTPacket> *> rt_vec{ &rt_consumer };
    MultiConsumer<RTPacket> rt_cons(rt_vec);
    rt_pl_ = new Pipeline<RTPacket>(rt_prod, rt_cons, "RTPacket", *notifier);
    
    rt_commander_ = factory_.getCommander(rt_transmit_stream_);
  }

  URStream rt_transmit_stream_;

  Pipeline<RTPacket> *rt_pl_;

  std::unique_ptr<URCommander> rt_commander_;

  void automaticErrorRecovery() {
    // TODO(jacky) this hasn't been implemented on UR5e
    throw "Automatic Error Recovery hasn't been implemented on UR5e!";
  }

 private:
  const int UR_RT_TRANSMIT_PORT_ = 30003;
  const int UR_RT_RECEIVE_PORT_ = 30013;

  URFactory factory_;
  URStream rt_receive_stream_;

};

#endif  // IAM_ROBOLIB_ROBOTS_UR5E_ROBOT_H_