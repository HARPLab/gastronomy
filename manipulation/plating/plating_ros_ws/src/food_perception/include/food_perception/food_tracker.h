#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "food_perception/project_pixel.h"
#include "food_perception/identify_food_pixel.h"
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PolygonStamped.h>
#include <memory>
#include <vector>

class FoodTracker
{
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber sub_;
    image_transport::Publisher mask_pub_;
    tf::TransformListener tf_listener_;
    ros::Publisher poly_pub_;
    std::vector<ros::Publisher> food_loc_pubs_;
    std::shared_ptr<PixelProjector> pix_proj_;
    std::shared_ptr<FoodPixelIdentifier> pix_identifier_;
    std::string camera_frame_, plane_frame_, negative_img_filename_;
    std::vector<std::string> positive_img_filenames_;
    std::vector<geometry_msgs::Point> *table_polygon_of_interest_;
    bool active_ = false;
  public:
    FoodTracker(std::string image_topic, std::string plane_frame, std::vector<std::string> positive_img_filenames, std::string negative_img_filename, std::vector<geometry_msgs::Point> *table_polygon_of_interest = NULL);
    void StartTracking();
    void imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
};


