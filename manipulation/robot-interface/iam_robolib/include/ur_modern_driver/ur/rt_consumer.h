#pragma once

#include <cstdlib>
#include <vector>

#include "ur_modern_driver/ur/consumer.h"

class RTConsumer : public URRTPacketConsumer
{
private:
  bool temp_only_;

  bool saveJoints(RTShared& packet);
  bool saveWrench(RTShared& packet);
  bool saveTool(RTShared& packet);
  bool saveTransform(RTShared& packet);
  bool saveTemperature(RTShared& packet);

  bool save(RTShared& packet);

public:
  RTConsumer(bool temp_only = false)
    : temp_only_(temp_only)
  {
  }

  virtual bool consume(RTState_V1_6__7& state);
  virtual bool consume(RTState_V1_8& state);
  virtual bool consume(RTState_V3_0__1& state);
  virtual bool consume(RTState_V3_2__3& state);
  virtual bool consume(RTState_V3_5__5_1& state);

  virtual void setupConsumer()
  {
  }
  virtual void teardownConsumer()
  {
  }
  virtual void stopConsumer()
  {
  }
};
