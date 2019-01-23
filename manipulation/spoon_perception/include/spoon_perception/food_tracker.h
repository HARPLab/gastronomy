#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "spoon_perception/project_pixel.h"
#include "spoon_perception/identify_food_pixel.h"
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <memory>

class FoodTracker
{
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber sub_;
    tf::TransformListener tf_listener_;
    ros::Publisher food_loc_pub_;
    std::shared_ptr<PixelProjector> pix_proj_;
    std::shared_ptr<FoodPixelIdentifier> pix_identifier_;
    std::string camera_frame_;
    std::string plane_frame_;
    bool active_ = false;
  public:
    FoodTracker(std::string image_topic, std::string plane_frame);
    void StartTracking();
    void imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
};


