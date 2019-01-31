#include "ur_modern_driver/ur/commander.h"
#include "ur_modern_driver/log.h"

bool URCommander::write(const std::string &s)
{
  size_t len = s.size();
  const uint8_t *data = reinterpret_cast<const uint8_t *>(s.c_str());
  size_t written;
  return stream_.write(data, len, written);
}

void URCommander::formatArray(std::ostringstream &out, std::array<double, 6> &values)
{
  std::string mod("[");
  for (auto const &val : values)
  {
    out << mod << val;
    mod = ",";
  }
  out << "]";
}

bool URCommander::uploadProg(const std::string &s)
{
  LOG_DEBUG("Sending program [%s]",s.c_str());
  return write(s);
}

bool URCommander::setToolVoltage(uint8_t voltage)
{
  if (voltage != 0 || voltage != 12 || voltage != 24)
    return false;

  std::ostringstream out;
  out << "set_tool_voltage(" << (int)voltage << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::setFlag(uint8_t pin, bool value)
{
  std::ostringstream out;
  out << "set_flag(" << (int)pin << "," << (value ? "True" : "False") << ")\n";
  std::string s(out.str());
  return write(s);
}
bool URCommander::setPayload(double value)
{
  std::ostringstream out;
  out << "set_payload(" << std::fixed << std::setprecision(5) << value << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::setAnalogOut(uint8_t pin, double value)
{
  std::ostringstream out;
  out << "sec io_fun():\n" << "set_standard_analog_out(" << (int)pin << "," << std::fixed << std::setprecision(5) << value << ")\n" << "end\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::setDigitalOut(uint8_t pin, bool value)
{
  std::ostringstream out;
  std::string func;

  if (pin < 8)
  {
    func = "set_standard_digital_out";
  }
  else if (pin < 16)
  {
    func = "set_configurable_digital_out";
    pin -= 8;
  }
  else if (pin < 18)
  {
    func = "set_tool_digital_out";
    pin -= 16;
  }
  else
    return false;

  out << "sec io_fun():\n" << func << "(" << (int)pin << "," << (value ? "True" : "False") << ")\n" << "end\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::movej(std::array<double, 6> &joint_positions, double joint_acceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "movej(";
  formatArray(out, joint_positions);
  out << ", a=" << joint_acceleration << ", v=" << 1.05 << ", t=" << 0.002 << ", r=" << 0 << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::movel(std::array<double, 6> &pose, double tool_acceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "movel(p";
  formatArray(out, pose);
  out << ", a=" << tool_acceleration << ", v=" << 0.25 << ", t=" << 0.002 << ", r=" << 0.0 << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::movep(std::array<double, 6> &pose, double tool_acceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "movel(p";
  formatArray(out, pose);
  out << ", a=" << tool_acceleration << ", v=" << 0.25 << ", r=" << 0.0 << ")\n";
  std::string s(out.str());
  return write(s);
}

// acceleration and velocity are not used
bool URCommander::servoj(std::array<double, 6> &joint_positions, double gain)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "servoj(";
  formatArray(out, joint_positions);
  out << ", 0, 0, " << 0.002 << ", 0.1," << gain << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::speedj(std::array<double, 6> &joint_speeds, double joint_acceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "speedj(";
  formatArray(out, joint_speeds);
  out << "," << joint_acceleration << "," << 0.002 << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::speedl(std::array<double, 6> &tool_speeds, double tool_acceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "speedl(";
  formatArray(out, tool_speeds);
  out << "," << tool_acceleration << "," << 0.002 << ")\n";
  std::string s(out.str());
  return write(s);
}

bool URCommander::stopj(double joint_deacceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "stopj(" << joint_deacceleration << ")\n";
  std::string s(out.str());
  return write(s);
}


bool URCommander::stopl(double tool_deacceleration)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(5);
  out << "stopl(" << tool_deacceleration << ")\n";
  std::string s(out.str());
  return write(s);
}