#include "food_perception/identify_food_pixel.h"
#include <iostream>

// this is allowed to modify binary_image
// find the food pixel center in the binary image
// return true if out parameter "pixel" is filled out
bool GetPixel(cv::Mat &binary_image, cv::Point2i &pixel)
{
  // minimum number of pixels in order to count as a splotch
  int min_num_pixels = 20;

  // https://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
  // erode anything with a radius less than 10 pixels
  int erosion_size = 2;
  cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                     cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                     cv::Point2i( erosion_size, erosion_size ) );
  cv::erode(binary_image, binary_image, element);

  // dilate to connect anything within 10 pixels (plus 10 pixels to undo the erode)
  int dilation_size = 5;
  element = cv::getStructuringElement( cv::MORPH_RECT,
                                     cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                     cv::Point2i( dilation_size, dilation_size ) );
  cv::dilate(binary_image, binary_image, element);
  
  // only return the maximal-area connected element
  // https://stackoverflow.com/questions/29108270/opencv-2-4-10-bwlabel-connected-components/30265609#30265609
  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(binary_image, labels, stats, centroids);
  
  if (nLabels <= 1)
  {
    // no object detected
    return false;
  }

  // index 0 is background
  // but it's still counted as part of nLabels...
  int largest_component = 1;
  int max_area = stats.at<int>(1, cv::CC_STAT_AREA);
  for (int i = 1; i < nLabels; i++)
  {
    int comp_area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (comp_area > max_area)
    {
      max_area = comp_area;
      largest_component = i;
    } 
  }

  if (max_area < min_num_pixels)
  {
    // object still too small 
    return false;
  }
  
  
  int x_center = centroids.at<double>(largest_component, 0);
  int y_center = centroids.at<double>(largest_component, 1);
  pixel.x = x_center;
  pixel.y = y_center;
  return true;
} 

FoodPixelIdentifier::FoodPixelIdentifier(std::vector<std::string> positive_img_filenames, std::string negative_img_filename) : it_(nh_)
{
  int i = 0;
  for (std::vector<std::string>::iterator itr = positive_img_filenames.begin(); itr != positive_img_filenames.end(); itr++)
  { 
    cv::Mat positive = cv::imread(*itr, CV_LOAD_IMAGE_COLOR);
    cv::Mat positive_vec = positive.reshape(3,positive.cols*positive.rows).reshape(1);
    positive_vecs_.push_back(positive_vec);
    image_transport::Publisher raw_pixels_pub_ = it_.advertise("raw_food_mask" + std::to_string(i),1);
    raw_pixels_pubs_.push_back(raw_pixels_pub_);
    i++;
  }
  cv::Mat negative = cv::imread(negative_img_filename, CV_LOAD_IMAGE_COLOR);
  cv::Mat negative_vec = negative.reshape(3,negative.cols*negative.rows).reshape(1);
  negative_vecs_.push_back(negative_vec); 
}

// given an input image and positive and negative examples of the food
// and (optionally) a mask
// fill in the pixels vector (out parameter) with all the centers of the food
// that lie within the mask
std::vector<bool> FoodPixelIdentifier::GetFoodPixelCenter(const cv::Mat &image, 
         std::vector<cv::Point2i> &pixels, 
         cv::Mat *mask)
{
  // avoid the computational complexity of overly large images
  int scaled_size_x = 200, scaled_size_y = 200;
  int smaller_image_x, smaller_image_y;

  cv::Mat scaledImage;
  bool resized = false;
  if (image.rows > scaled_size_x || image.cols > scaled_size_y)
  {
    cv::resize(image, scaledImage, cv::Size(scaled_size_y, scaled_size_x),0,0);
    resized = true;
    smaller_image_x = scaled_size_x;
    smaller_image_y = scaled_size_y;
  }
  else
  {
    scaledImage = image;
    scaled_size_x = image.rows;
    scaled_size_y = image.cols;
  }

  cv::Mat image_vec = scaledImage.reshape(3,scaled_size_x * scaled_size_y).reshape(1);

  std::vector<cv::Mat> min_dist_positives;
  for (std::vector<cv::Mat>::iterator itr = positive_vecs_.begin(); itr != positive_vecs_.end(); itr++)
  {
    cv::Mat dist_positive;
    cv::batchDistance(image_vec, *itr, dist_positive, -1, cv::noArray());
    cv::Mat min_dist_positive;
    cv::reduce(dist_positive, min_dist_positive, 1, cv::REDUCE_MIN);
    min_dist_positives.push_back(min_dist_positive);
  }
  
  cv::Mat dist_negative; 
  cv::batchDistance(image_vec, negative_vecs_[0], dist_negative, -1, cv::noArray());
  cv::Mat min_dist_negative;
  cv::reduce(dist_negative, min_dist_negative, 1, cv::REDUCE_MIN);

  // New strategy: find min distance across all options, then check if
  // our array is equal to that min distance. Treats equality slightly differently
  // than originally planned (could fix with some epsilon if we wanted to)
  // but makes computation smooth and painless
  cv::Mat min_dist_cumulative = min_dist_negative;
  for (std::vector<cv::Mat>::iterator itr = min_dist_positives.begin(); itr != min_dist_positives.end(); itr++)
  {
    cv::min(*itr, min_dist_cumulative, min_dist_cumulative);
  }
    
  std::vector<bool> success_vec;
  int i = 0;
  // Generate binary image, find food point, and save it off for each type of food
  for (std::vector<cv::Mat>::iterator itr = min_dist_positives.begin(); itr != min_dist_positives.end(); itr++)
  {
    cv::Mat binary_image;
    cv::compare(*itr, min_dist_cumulative, binary_image, cv::CMP_EQ);
    binary_image = binary_image.reshape(1,scaled_size_x);
    
    cv::Mat binary_image_unscaled; 
    if (resized) {
      // http://answers.opencv.org/question/68507/cvresize-incorrect-interpolation-with-inter_nearest/ 
      cv::resize(binary_image, binary_image_unscaled, cv::Size(image.cols,image.rows),0,0, cv::INTER_NEAREST);
    } else {
      binary_image_unscaled = binary_image;
    }

    // apply the mask to the binary "relevant pixels" image 
    if (mask) {
      binary_image_unscaled = mask->mul(binary_image_unscaled);
    }

    // for debugging, publish the image pixels to a ros topic
    std_msgs::Header head; 
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(head, "mono8", binary_image_unscaled* 255).toImageMsg();
    raw_pixels_pubs_[i].publish(msg); 
    i++;

    cv::Point2i pixel;
    bool success = GetPixel(binary_image_unscaled, pixel);
    success_vec.push_back(success);
    pixels.push_back(pixel);
  }
  return(success_vec);
}
