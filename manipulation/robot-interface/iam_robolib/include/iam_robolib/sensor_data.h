#ifndef IAM_ROBOLIB_SENSOR_DATA_H_
#define IAM_ROBOLIB_SENSOR_DATA_H_

class SensorData {
 public:
  explicit SensorData(float *v) : values_{v} {};

  /**
   * Parse data from memory.
   */
  virtual void parse_data() = 0;

  /**
   * Initialize any other hand.
   */
  virtual void initialize_data() = 0;

  /**
   * Should we terminate the current skill.
   */
  virtual bool update_data() = 0;

 protected:
  float *values_=0;
};

#endif  // IAM_ROBOLIB_SENSOR_DATA_H_