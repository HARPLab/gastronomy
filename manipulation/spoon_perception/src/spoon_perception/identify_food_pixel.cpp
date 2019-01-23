#include "spoon_perception/identify_food_pixel.h"
#include <iostream>
#include <ros/ros.h>

bool FoodPixelIdentifier::GetFoodPixelCenter(const cv::Mat &image, cv::Point2d &pixel)
{
  //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  //cv::imshow( "Display window", image );     
  //cv::waitKey(0);

  // minimum number of pixels in order to count as a tomato
  int min_num_pixels = 1000;
  
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

  cv::Mat tomato = cv::imread("/home/helvellyn/vision_chefbot_ws/src/spoon_perception/test/input_data/JustTomato.jpg", CV_LOAD_IMAGE_COLOR);
  cv::Mat noTomato = cv::imread("/home/helvellyn/vision_chefbot_ws/src/spoon_perception/test/input_data/NoTomato.jpg", CV_LOAD_IMAGE_COLOR);
  //std::cout << image.cols << "," << image.rows << " I guess\n";
  //std::cout << scaledImage.cols << "," << scaledImage.rows << " I guess\n";
  //std::cout << tomato.cols << "," << tomato.rows << " I guess\n";
  //std::cout << noTomato.cols << "," << noTomato.rows << " I guess\n";

  cv::Mat image_vec = scaledImage.reshape(3,scaled_size_x * scaled_size_y).reshape(1);
  cv::Mat tomato_vec = tomato.reshape(3,tomato.cols*tomato.rows).reshape(1);
  cv::Mat noTomato_vec = noTomato.reshape(3,noTomato.cols*noTomato.rows).reshape(1);

  //std::cout << image_vec.cols << "," << image_vec.rows << " I guess\n";
  //std::cout << tomato_vec.cols << "," << tomato_vec.rows << " I guess\n";
  //std::cout << noTomato_vec.cols << "," << noTomato_vec.rows << " I guess\n";

  cv::Mat dist_tomato, dist_noTomato; 
  cv::batchDistance(image_vec, tomato_vec, dist_tomato, -1, cv::noArray());
  cv::batchDistance(image_vec, noTomato_vec, dist_noTomato, -1, cv::noArray());

  cv::Mat min_dist_tomato, min_dist_noTomato;
  cv::reduce(dist_tomato, min_dist_tomato, 1, cv::REDUCE_MIN);
  cv::reduce(dist_noTomato, min_dist_noTomato, 1, cv::REDUCE_MIN);
  

  cv::Mat binary_location;
  cv::compare(min_dist_tomato, min_dist_noTomato, binary_location, cv::CMP_LE);
  binary_location = binary_location.reshape(1,scaled_size_x);
  
  cv::Mat binary_location_unscaled; 
  if (resized) {
    cv::resize(binary_location, binary_location_unscaled, cv::Size(image.cols,image.rows),0,0);
  } else {
    binary_location_unscaled = binary_location;
  }

  
  cv::Moments moments = cv::moments(binary_location_unscaled, true);
  int x_center = moments.m10/moments.m00;
  int y_center = moments.m01/moments.m00;
  std::cout << "x_center: " << x_center << "; y_center: " << y_center << "; count: " << moments.m00 << "\n";

  if (moments.m00 < min_num_pixels)
  {
    // no tomato detected
    return false;
  }
  
  pixel.x = x_center;
  pixel.y = y_center;
  //cv::circle(binary_location_unscaled,pixel,10,cv::Scalar( 0, 0, 255 ));
  //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  //cv::imshow( "Display window", binary_location_unscaled);     
  //cv::waitKey(0);
  return true;
}
