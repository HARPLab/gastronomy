#include "food_perception/food_tracker.h"

// Params:
// * image_topic: the ros topic streaming camera images
// * plane_frame: the table plane (z=0) we project food pixels onto
// * table_polygon_of_interest: a vector of points defining a polygon of interest (ignore food outside this polygon). All points should have z = 0 (should lie on table).
FoodTracker::FoodTracker(std::string image_topic, std::string plane_frame, std::vector<std::string> positive_img_filenames, std::string negative_img_filename, std::vector<geometry_msgs::Point> *table_polygon_of_interest) : it_(nh_), plane_frame_(plane_frame), table_polygon_of_interest_(table_polygon_of_interest), positive_img_filenames_(positive_img_filenames), negative_img_filename_(negative_img_filename)
{  
  ROS_WARN("[food_tracker] subscribing to camera");
  sub_ = it_.subscribeCamera(image_topic, 1, &FoodTracker::imageCb, this);
  ROS_WARN("[food_tracker] advertizing food topics");
  for (int i = 0; i < positive_img_filenames_.size(); i++)
  {
    ros::Publisher food_loc_pub = nh_.advertise<geometry_msgs::PointStamped>("food" + std::to_string(i),1);
    food_loc_pubs_.push_back(food_loc_pub); 
  }
  ROS_WARN("[food_tracker] advertizing (debugging) polygon topic");
  poly_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("food_filter_polygon",1);
  mask_pub_ = it_.advertise("mask",1);
}

void FoodTracker::StartTracking()
{
  active_ = true;
}

// the image_msg and info_msg should have "close" timestamps, but just in case,
// we solely use image_msg.header.stamp when checking the time to use for tf
void FoodTracker::imageCb(const sensor_msgs::ImageConstPtr& image_msg,
             const sensor_msgs::CameraInfoConstPtr& info_msg)
{
  if (!pix_proj_)
  {
    ROS_WARN("[food_tracker] Initializing pixel projector");
    camera_frame_ = info_msg->header.frame_id;
    std::shared_ptr<PixelProjector> pix_proj(new PixelProjector(*info_msg, camera_frame_, plane_frame_)); 
    pix_proj_ = pix_proj;
  }
  
  if (!pix_identifier_)
  {
    ROS_WARN("[food_tracker] Initializing pixel identifier");
    std::shared_ptr<FoodPixelIdentifier> pix_identifier(new FoodPixelIdentifier(positive_img_filenames_, negative_img_filename_));
    pix_identifier_ = pix_identifier;
  }

  // give some time for the newly initialized tf listener to hear the tf transform
  if (!active_)
  {
    ROS_WARN("[food_tracker] Waiting because not active yet...");
    return;
  }
  
  // give some time for the newly initialized tf listener to hear the tf transform
  if (!pix_proj_->CanProject(image_msg->header.stamp))
  {
    ROS_WARN("[food_tracker] Waiting because pixel projector not yet ready to transform this timestamp...");
    return;
  }

  cv::Mat image;
  cv_bridge::CvImagePtr input_bridge;
  try {
    input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    image = input_bridge->image;
  }
  catch (cv_bridge::Exception& ex){
    ROS_ERROR("[food_tracker] Failed to convert image");
    return;
  }

  cv::Mat *mask_pointer = NULL; 
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);

  // If we have a region of interest,
  // generate a mask so we only look at pixels that fall inside that polygon
  if (table_polygon_of_interest_)
  {
    geometry_msgs::PointStamped stamped_vertex;
    // fill out point header to make a template for a stamped point 
    stamped_vertex.header.stamp = image_msg->header.stamp;
    stamped_vertex.header.frame_id = plane_frame_;
    std::vector<cv::Point2i> image_filter_vertices;
    geometry_msgs::PolygonStamped polygon_msg; 
    polygon_msg.header.stamp = image_msg->header.stamp;
    polygon_msg.header.frame_id = plane_frame_;
    for (int i = 0; i < table_polygon_of_interest_->size(); i++)
    {
      stamped_vertex.point = table_polygon_of_interest_->at(i);
      //gotta convert to Point32...
      geometry_msgs::Point32 vertex;
      vertex.x = stamped_vertex.point.x; 
      vertex.y = stamped_vertex.point.y; 
      vertex.z = stamped_vertex.point.z; 
      polygon_msg.polygon.points.push_back(vertex);
      // the following converts to integer, which I'm fine with
      cv::Point2i image_filter_vertex;
      bool is_in_front_of_camera = pix_proj_->PointStampedProjectedToPixel(stamped_vertex, image_filter_vertex);
      if (!is_in_front_of_camera)
      {
        //ROS_WARN("Ignoring this image, because at least one point of polygon is behind camera");
        return;
      }
      image_filter_vertices.push_back(image_filter_vertex);
    }

    // https://stackoverflow.com/questions/43443127/opencv-how-to-create-a-mask-in-the-shape-of-a-polygon
    cv::fillConvexPoly(mask, image_filter_vertices.data(), image_filter_vertices.size(), cv::Scalar(1));

    poly_pub_.publish(polygon_msg);
    std_msgs::Header mask_head = image_msg->header;
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(mask_head, "mono8", mask * 255).toImageMsg();
    mask_pub_.publish(msg);
    mask_pointer = &mask;
  }

  std::vector<cv::Point2i> food_pixels;
  std::vector<std::string> negative_img_filenames;
  negative_img_filenames.push_back(negative_img_filename_);
  std::vector<bool> success_vec = pix_identifier_->GetFoodPixelCenter(image, food_pixels, mask_pointer);

  for (int i = 0; i < success_vec.size(); i++)
  {
    if (!success_vec[i])
    {
      //ROS_WARN("[food_tracker] food %d not seen", i);
      continue;
    }
    
    //ROS_WARN("[food_tracker] publishing food %d", i);
    cv::Point2i food_pixel = food_pixels[i];
    geometry_msgs::PointStamped food_loc_msg = pix_proj_->PixelProjectedOnXYPlane(food_pixel, image_msg->header.stamp);
    food_loc_pubs_[i].publish(food_loc_msg);
  }
}

