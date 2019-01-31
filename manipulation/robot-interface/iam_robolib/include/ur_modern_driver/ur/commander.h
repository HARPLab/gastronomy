#pragma once
#include <array>
#include <iomanip>
#include <sstream>
#include "ur_modern_driver/ur/stream.h"

class URCommander
{
private:
  URStream &stream_;

protected:
  bool write(const std::string &s);
  void formatArray(std::ostringstream &out, std::array<double, 6> &values);

public:
  URCommander(URStream &stream) : stream_(stream)
  {
  }

  // shared
  bool uploadProg(const std::string &s);
  bool setToolVoltage(uint8_t voltage);
  bool setFlag(uint8_t pin, bool value);
  bool setPayload(double value);

  bool setDigitalOut(uint8_t pin, bool value);
  bool setAnalogOut(uint8_t pin, double value);

  bool movej(std::array<double, 6> &joint_positions, double joint_acceleration);
  bool movel(std::array<double, 6> &pose, double tool_acceleration);
  bool movep(std::array<double, 6> &pose, double tool_acceleration);
  bool servoj(std::array<double, 6> &joint_positions, double gain);
  bool speedj(std::array<double, 6> &joint_speeds, double joint_acceleration);
  bool speedl(std::array<double, 6> &tool_speeds, double tool_acceleration);
  bool stopj(double joint_deacceleration = 10.0);
  bool stopl(double tool_deacceleration = 10.0);
};
