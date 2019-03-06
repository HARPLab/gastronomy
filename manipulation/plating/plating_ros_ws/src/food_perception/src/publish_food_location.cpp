#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "food_perception/food_tracker.h"
#include "food_perception/param_parser.h"
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/algorithm/string.hpp>

int main(int argc, char** argv)
{
  ROS_WARN("Starting main method");
  ros::init(argc, argv, "food_publisher");
  ros::NodeHandle nh;
  std::string camera_topic, table_frame_id, roi_polygon_raw_str;
  std::string positive_img_filenames_raw, negative_img_filename;
  ros::param::get("~camera_topic", camera_topic); 
  ros::param::get("~table_frame_id", table_frame_id); 
  ros::param::get("~positive_img_filenames", positive_img_filenames_raw);
  std::vector<std::string> positive_img_filenames;
  if (!ParseFilenamesParam(positive_img_filenames_raw, positive_img_filenames))
  {
    ROS_ERROR("Error parsing positive_img_filenames parameter. That parameter should contain paths to files containing positive object images, with names separated by only a comma and no extra spaces.");
    throw;
  }
  ros::param::get("~negative_img_filename", negative_img_filename); 
  // this is the syntax to include a default value
  ros::param::param<std::string>("~roi_polygon", roi_polygon_raw_str, ""); 
  std::vector<geometry_msgs::Point> *poly_of_interest_ptr = NULL; 
  std::vector<geometry_msgs::Point> poly_of_interest;
  if (roi_polygon_raw_str.size() > 0)
  {
    if (!ParsePolygonParam(roi_polygon_raw_str, poly_of_interest))
    {
      ROS_ERROR("Error parsing roi_polygon parameter. That parameter (if included) should be a string of the form '(x0,y0), (x1, y1), ..., (xn, yn)'");
      throw;
    }
    poly_of_interest_ptr = &poly_of_interest;
  }
  std::string image_topic = nh.resolveName(camera_topic);
  FoodTracker foodTracker(image_topic, table_frame_id, positive_img_filenames, negative_img_filename, poly_of_interest_ptr);
  ROS_WARN("Sleeping for a spell");
  for (int i=0; i < 10; i++)
  {
    ros::Duration(0.1).sleep();
    ros::spinOnce();
  }
  foodTracker.StartTracking();
  ros::spin();
}
