#include <spoon_perception/project_pixel.h>

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

PixelProjector::PixelProjector(const sensor_msgs::CameraInfo &camera_info, std::string camera_frame, std::string plane_frame) : camera_frame_(camera_frame), plane_frame_(plane_frame), camera_info_(camera_info)
{
  cam_model_.fromCameraInfo(camera_info_);
  // wait for the tf to load up 
  ros::Duration timeout(1.0);
  tf_listener_.waitForTransform(plane_frame_, camera_frame_, ros::Time(0), timeout);
}

geometry_msgs::PointStamped PixelProjector::PixelProjectedOnXYPlane(const cv::Point2d & uv_rect, const ros::Time acquisition_time)
{
  cv::Point3d pixel_ray = cam_model_.projectPixelTo3dRay(uv_rect);
  tf::Vector3 pixel_ray_camera_frame = tf::Vector3(pixel_ray.x, pixel_ray.y, pixel_ray.z);
  std::string camera_frame = cam_model_.tfFrame();
  
  tf::StampedTransform transform;
  ros::Duration timeout(1.0 / 30);
  try {
    tf_listener_.waitForTransform(plane_frame_, camera_frame, acquisition_time, timeout);
    tf_listener_.lookupTransform(plane_frame_, camera_frame, acquisition_time, transform);
  }
  catch (tf::TransformException& ex) {
    ROS_ERROR("[project_pixel] TF exception:\n%s", ex.what());
    throw;
  }

  geometry_msgs::PointStamped stamped_point = RayProjectedOnXYPlane(pixel_ray_camera_frame, transform);

  return stamped_point;
}
