#include "iam_robolib/sensor_data_manager.h"


#include <google/protobuf/message.h>


SensorDataManagerReadStatus SensorDataManager::readJointSensorInfoMessage(JointSensorInfo &message) {
   std::function<bool(const void *bytes, int data_size)>
       parse_callback = [&](const void *bytes, int data_size) -> bool {
     // get state variables
     return message.ParseFromArray(bytes, data_size);
   };
   return readMessageAsBytes(parse_callback);
}

SensorDataManagerReadStatus SensorDataManager::readMessageAsBytes(std::function< bool(const void *bytes, int data_size)> parse_callback) {
  SensorDataManagerReadStatus status;

  // std::cout << "Try to read message" << std::endl;
  try {
    if (buffer_mutex_->try_lock()) {
      int has_new_message = static_cast<int>(buffer_[0]);
      if (has_new_message == 1) {
        int sensor_msg_type = static_cast<int>(buffer_[1]);

        int data_size = (buffer_[2] + (buffer_[3] << 8) + (buffer_[4] << 16) + (buffer_[5] << 24));

        // std::cout << "Will try to read sensor message of type: " << sensor_msg_type << " size: " << data_size << std::endl;
        if (parse_callback(buffer_ + 6, data_size)) {

          status = SensorDataManagerReadStatus::SUCCESS;
          std::cout << "Did read successfully from buffer" << std::endl;

          // Update the first value in the buffer to 0 (implying that we read it correctly?)
          buffer_[0] = 0;

        } else {
          status = SensorDataManagerReadStatus::FAIL_TO_READ;
          std::cout << "Protobuf failed to read message from memory" << std::endl;
        }


      } else {
        // std::cout << "No new sensor message" << std::endl;
        status = SensorDataManagerReadStatus::NO_NEW_MESSAGE;
      }

      buffer_mutex_->unlock();
    }
  } catch (boost::interprocess::lock_exception) {
    status = SensorDataManagerReadStatus::FAIL_TO_GET_LOCK;
    // Do nothing
  }

  return status;

}
