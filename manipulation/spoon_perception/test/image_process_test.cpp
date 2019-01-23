// see http://wiki.ros.org/gtest
// Bring in my package's API, which is what I'm testing
#include "spoon_perception/identify_food_pixel.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

// Declare a test
TEST(TestSuite, test_find_red)
{
  cv::Mat image;
  image = cv::imread("/home/helvellyn/vision_chefbot_ws/src/spoon_perception/test/input_data/TestImage.jpg", CV_LOAD_IMAGE_COLOR);
  FoodPixelIdentifier food_identifier;
  cv::Point2d point = food_identifier.GetFoodPixelCenter(image);
  std::cout << "point" << point.x << "," << point.y << "\n";
}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
