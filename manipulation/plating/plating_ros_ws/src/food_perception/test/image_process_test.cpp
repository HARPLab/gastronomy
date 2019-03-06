// see http://wiki.ros.org/gtest
// Bring in my package's API, which is what I'm testing
#include "food_perception/identify_food_pixel.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <vector>

// Declare a test
TEST(TestSuite, test_find_red)
{
  std::string package_path = ros::package::getPath("food_perception");
  cv::Mat image = cv::imread(package_path + "/test/input_data/MosaicView.jpg", CV_LOAD_IMAGE_COLOR);
  std::string tomato = package_path + "/test/input_data/SmallRed.png";
  std::string noTomato = package_path + "/test/input_data/SmallBlue.png";
  std::vector<std::string> tomatos;
  tomatos.push_back(tomato);
  
  FoodPixelIdentifier food_identifier(tomatos, noTomato);
  std::vector<cv::Point2i> points;
  std::vector<bool> found = food_identifier.GetFoodPixelCenter(image, points);
  //std::cout << "point" << point.x << "," << point.y << "\n";
}


// Test finding the center of the largest of several splotches 
TEST(TestSuite, test_find_center)
{
  cv::Mat image = cv::Mat::zeros(100, 100, CV_8U);
  cv::Point2i point1(30, 30), point2(70,70);
  // negative line width means filled
  cv::circle(image, point1, 10, cv::Scalar(1), -1);
  cv::circle(image, point2, 6, cv::Scalar(1), -1);
  
  //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  //cv::imshow( "Display window", image * 255);     
  //cv::waitKey(0);

  cv::Point2i pixel;
  ASSERT_TRUE(GetPixel(image, pixel));

  //simple regression test:
  EXPECT_EQ(30, pixel.x);
  EXPECT_EQ(30, pixel.y);
}

TEST(TestSuite, test_mask)
{
  std::string package_path = ros::package::getPath("food_perception");
  cv::Mat image = cv::imread(package_path + "/test/input_data/MosaicView.jpg", CV_LOAD_IMAGE_COLOR);
  std::string tomato = package_path + "/test/input_data/SmallRed.png";
  std::string noTomato = package_path + "/test/input_data/SmallBlue.png";
  std::vector<std::string> tomatos;
  tomatos.push_back(tomato);
  
  FoodPixelIdentifier food_identifier(tomatos, noTomato);

  // make a mask
  std::vector<cv::Point2i> image_filter_vertices;
  image_filter_vertices.push_back(cv::Point2i(image.cols/3,image.rows/3));
  image_filter_vertices.push_back(cv::Point2i(image.cols/3,2 * image.rows/3));
  image_filter_vertices.push_back(cv::Point2i(2*image.cols/3,2 * image.rows/3));
  image_filter_vertices.push_back(cv::Point2i(2*image.cols/3,image.rows/3));
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
  cv::fillConvexPoly(mask, image_filter_vertices.data(), image_filter_vertices.size(), cv::Scalar(1));

  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Display window", mask * 255);     
  //cv::imshow( "Display window", image);     
  cv::waitKey(0);
  
  std::vector<cv::Point2i> points;
  std::vector<bool> found = food_identifier.GetFoodPixelCenter(image, points, &mask);
  ASSERT_TRUE(found[0]);
  std::cout << "point" << points[0].x << "," << points[0].y << "\n";

  //cv::circle(image,points[0],10,cv::Scalar( 0, 0, 255 ));
  //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
  //cv::imshow( "Display window", image);     
  //cv::waitKey(0);

  //simple regression test:
  cv::Point2i point = points[0];
  EXPECT_EQ(489, point.x);
  EXPECT_EQ(383, point.y);
}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
