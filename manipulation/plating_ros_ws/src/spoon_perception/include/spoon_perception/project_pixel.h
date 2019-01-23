#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv/cv.h>

geometry_msgs::PointStamped RayProjectedOnXYPlane(tf::Vector3 &ray_camera_frame, tf::StampedTransform transform);

class PixelProjector
{
  private:
    ros::NodeHandle nh_;
    tf::TransformListener tf_listener_;
    image_geometry::PinholeCameraModel cam_model_;
    std::string camera_frame_;
    std::string plane_frame_;
    sensor_msgs::CameraInfo camera_info_;

  public:
    PixelProjector(const sensor_msgs::CameraInfo &camera_info, std::string camera_frame, std::string plane_frame);
    geometry_msgs::PointStamped PixelProjectedOnXYPlane(const cv::Point2d & uv_rect, const ros::Time acquisition_time);
};
