#include <ros/ros.h>
#include <ros/package.h>
#include <fstream>
#include "yaml-cpp/yaml.h"

#include <interactive_markers/interactive_marker_server.h>
#include <tf/transform_datatypes.h>

using namespace visualization_msgs;

std::string _config_path;

void processFeedback(
    const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback )
{
  ROS_INFO_STREAM( feedback->marker_name << " is now at "
      << feedback->pose.position.x << ", " << feedback->pose.position.y
      << ", " << feedback->pose.position.z );
  
  std::vector<double> q = {  
    feedback->pose.orientation.x, feedback->pose.orientation.y, feedback->pose.orientation.z, feedback->pose.orientation.w };
  std::vector<double> t = {feedback->pose.position.x, feedback->pose.position.y, feedback->pose.position.z};
  ros::param::set("camera_calib_params/QuaternionXYZW", q);
  ros::param::set("camera_calib_params/TranslationXYZ", t);
  std::ofstream myfile;
  myfile.open (_config_path + "calibrationParameters.yml");
  myfile << "QuaternionXYZW:\n";
  for (int i=0; i < q.size(); i++){
    myfile << "- " << q[i] << "\n";
  }
  myfile << "TranslationXYZ:\n";
  for (int i=0; i < t.size(); i++){
    myfile << "- " << t[i] << "\n";
  }
  myfile.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_marker");

  _config_path = ros::package::getPath("feedbot_trajectory_logic").append("/config/");
  
  YAML::Node calibParams = YAML::LoadFile(_config_path + "calibrationParameters.yml");
  std::vector<double> init_quat = calibParams["QuaternionXYZW"].as<std::vector<double>>();
  std::vector<double> init_trans = calibParams["TranslationXYZ"].as<std::vector<double>>();
  ros::param::set("camera_calib_params/QuaternionXYZW", init_quat); 
  ros::param::set("camera_calib_params/TranslationXYZ", init_trans);

  // create an interactive marker server on the topic namespace simple_marker
  interactive_markers::InteractiveMarkerServer server("simple_marker");

  // create an interactive marker for our server
  visualization_msgs::InteractiveMarker int_marker;
  std::string world_frame_name;
  ros::param::get("~world_frame_name", world_frame_name);
  int_marker.header.frame_id = world_frame_name;
  int_marker.header.stamp=ros::Time::now();
  int_marker.name = "my_marker";
  int_marker.description = "Simple Control";

  // create a grey box marker
  visualization_msgs::Marker box_marker;
  box_marker.type = visualization_msgs::Marker::CUBE;
  box_marker.pose.orientation.x = init_quat[0];
  box_marker.pose.orientation.y = init_quat[1];
  box_marker.pose.orientation.z = init_quat[2];
  box_marker.pose.orientation.w = init_quat[3];
  box_marker.pose.position.x = init_trans[0];
  box_marker.pose.position.y = init_trans[1];
  box_marker.pose.position.z = init_trans[2];
  box_marker.scale.x = 0.25;
  box_marker.scale.y = 0.25;
  box_marker.scale.z = 0.25;
  box_marker.color.r = 0.5;
  box_marker.color.g = 0.5;
  box_marker.color.b = 0.5;
  box_marker.color.a = 0.0;

  // create a non-interactive control which contains the box
  visualization_msgs::InteractiveMarkerControl box_control;
  box_control.orientation_mode = InteractiveMarkerControl::FIXED;
  box_control.always_visible = true;
  box_control.markers.push_back( box_marker );

  // add the control to the interactive marker
  int_marker.controls.push_back( box_control );

  // create a control which will move the box
  // this control does not contain any markers,
  // which will cause RViz to insert two arrows
  visualization_msgs::InteractiveMarkerControl control;
    control.orientation_mode = InteractiveMarkerControl::FIXED;
    control.orientation.w = 1;
    control.orientation.x = 1;
    control.orientation.y = 0;
    control.orientation.z = 0;
    control.name = "rotate_x";
    control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(control);
    control.name = "move_x";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);

    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 1;
    control.orientation.z = 0;
    control.name = "rotate_z";
    control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(control);
    control.name = "move_z";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);

    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 0;
    control.orientation.z = 1;
    control.name = "rotate_y";
    control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    int_marker.controls.push_back(control);
    control.name = "move_y";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
int_marker.controls.push_back(control);

  // add the interactive marker to our collection &
  // tell the server to call processFeedback() when feedback arrives for it
  server.insert(int_marker, &processFeedback);

  // 'commit' changes and send to all clients
  server.applyChanges();

  // start the ROS main loop
  ros::spin();
}
