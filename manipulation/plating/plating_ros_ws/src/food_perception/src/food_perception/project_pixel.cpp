#include <food_perception/project_pixel.h>

geometry_msgs::PointStamped RayProjectedOnXYPlane(tf::Vector3 &ray_camera_frame, tf::StampedTransform transform)
{
  tf::Point pixel_ray_origin = transform.getOrigin();
  tf::Quaternion rotation_quat(transform.getRotation());
  tf::Vector3 zero(0,0,0);
  tf::Transform rotation_transform(rotation_quat, zero);
  tf::Vector3 pixel_ray_plane_frame = rotation_transform * ray_camera_frame;
  // I'm not a huge fan of this, but here's how we do it.
  // First, save off the "normal distance" (which is the length
  // of the pixel_ray in the opposite direction of the vector normal to our intersection plane)
  float normal_ray_length = pixel_ray_plane_frame.z();
  float normal_ray_start_position = pixel_ray_origin.z();
  float normal_ray_multiplier = - normal_ray_start_position/normal_ray_length;
  double point_x = pixel_ray_origin.x() + normal_ray_multiplier * pixel_ray_plane_frame.x();
  double point_y = pixel_ray_origin.y() + normal_ray_multiplier * pixel_ray_plane_frame.y();
  geometry_msgs::Point point;
  point.x = point_x;
  point.y = point_y;
  point.z = 0.0; 
  std_msgs::Header header;
  header.stamp = transform.stamp_;
  header.frame_id = transform.frame_id_;
  geometry_msgs::PointStamped stamped_point;
  stamped_point.header = header;
  stamped_point.point = point;
  return stamped_point;
}

// if the stamp is before the earliest tf transform, there's no way we can transform
bool PixelProjector::CanProject(ros::Time stamp)
{
  return stamp > earliest_projectable_stamp_;
}

PixelProjector::PixelProjector(const sensor_msgs::CameraInfo &camera_info, std::string camera_frame, std::string plane_frame) : camera_frame_(camera_frame), plane_frame_(plane_frame), camera_info_(camera_info)
{
  cam_model_.fromCameraInfo(camera_info_);
  bool tf_loaded = false;
  // wait for the tf to load up for the first time
  while (ros::ok() && !tf_loaded)
  {
    ros::Duration timeout(1.0);
    try
    {
      ROS_WARN("Waiting for tf to load for first time");
      tf_listener_.waitForTransform(plane_frame_, camera_frame_, ros::Time(0), timeout);
      // if the line above throws, the lines below won't be run
      earliest_projectable_stamp_ = ros::Time::now(); 
      tf_loaded = true;
    }
    catch (...)
    {
      // do nothing. Tf's waitForTransform API seems to be error-based when waiting for the first transform...
    }
  }
}

geometry_msgs::PointStamped PixelProjector::PixelProjectedOnXYPlane(const cv::Point2d & uv_rect, const ros::Time acquisition_time)
{
  cv::Point3d pixel_ray = cam_model_.projectPixelTo3dRay(uv_rect);
  tf::Vector3 pixel_ray_camera_frame = tf::Vector3(pixel_ray.x, pixel_ray.y, pixel_ray.z);
  std::string camera_frame = cam_model_.tfFrame();
  
  tf::StampedTransform transform;
  ros::Duration timeout(1.0 / 10);
  try {
    tf_listener_.waitForTransform(plane_frame_, camera_frame, acquisition_time, timeout);
    tf_listener_.lookupTransform(plane_frame_, camera_frame, acquisition_time, transform);
  }
  catch (tf::TransformException& ex) {
    ROS_ERROR("[project_pixel] TF exception in PixelProjectedOnXYPlane:\n%s", ex.what());
    throw;
  }

  geometry_msgs::PointStamped stamped_point = RayProjectedOnXYPlane(pixel_ray_camera_frame, transform);

  return stamped_point;
}

// return false if the point is behind the camera (don't transform/publish if camera is pointing away from table
bool PixelProjector::PointStampedProjectedToPixel(const geometry_msgs::PointStamped point, cv::Point2i &pixel)
{
  geometry_msgs::PointStamped camera_frame_point;
  ros::Duration timeout(1.0 / 10);
  try {
    tf_listener_.waitForTransform(point.header.frame_id, camera_frame_, point.header.stamp, timeout);
    tf_listener_.transformPoint(camera_frame_, point, camera_frame_point);
  }
  catch (tf::TransformException& ex) {
    ROS_ERROR("[project_pixel] TF exception in PointStampedProjectedToPixel:\n%s", ex.what());
    throw;
  }

  // don't try to project the point if it's behind the camera. If any point is behind the camera, ignore this frame.
  if(camera_frame_point.point.z < 0)
  {
    return false;
  } 
  cv::Point3d cv_point3d(camera_frame_point.point.x, camera_frame_point.point.y, camera_frame_point.point.z);
  cv::Point2d cv_point2d = cam_model_.project3dToPixel(cv_point3d);
  // the following converts to integer. But that's ok!
  pixel.x = cv_point2d.x;
  pixel.y = cv_point2d.y;
  return true;
}
