#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "spoon_perception/food_tracker.h"
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

int main(int argc, char** argv)
{
  ROS_WARN("Starting main method");
  ros::init(argc, argv, "food_publisher");
  ros::NodeHandle nh;
  std::string camera_topic, table_frame_id;
  ros::param::get("~camera_topic", camera_topic); 
  ros::param::get("~table_frame_id", table_frame_id); 
  std::string image_topic = nh.resolveName(camera_topic);
  FoodTracker foodTracker(image_topic, table_frame_id);
  ROS_WARN("Sleeping for a spell");
  for (int i=0; i < 10; i++)
  {
    ros::Duration(0.1).sleep();
    ros::spinOnce();
  }
  foodTracker.StartTracking();
  ros::spin();
}
