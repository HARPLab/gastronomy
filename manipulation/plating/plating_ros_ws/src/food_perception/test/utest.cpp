// see http://wiki.ros.org/gtest
// Bring in my package's API, which is what I'm testing
#include "food_perception/project_pixel.h"
#include "food_perception/param_parser.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <math.h>

// Declare a test
TEST(TestSuite, test_projection)
{
  tf::Vector3 camera_ray(1,1,-1);
  tf::Quaternion quat(0,0,0,1); //x,y,z,w
  tf::Vector3 camera_loc(10,20,30);
  tf::Transform transform(quat, camera_loc);
  ros::Time stamp(33);
  tf::StampedTransform stamped_transform(transform, stamp, "table", "camera");
  geometry_msgs::PointStamped result_point = RayProjectedOnXYPlane(camera_ray, stamped_transform);
  EXPECT_EQ(40, result_point.point.x);
  EXPECT_EQ(50, result_point.point.y);
  EXPECT_EQ(0, result_point.point.z);
  
  tf::Quaternion quat2(0,0,sqrt(2),sqrt(2)); //x,y,z,w
  tf::Transform transform2(quat2, camera_loc);
  tf::StampedTransform stamped_transform2(transform2, stamp, "table", "camera");
  geometry_msgs::PointStamped result_point2 = RayProjectedOnXYPlane(camera_ray, stamped_transform2);
  EXPECT_FLOAT_EQ(-20, result_point2.point.x);
  EXPECT_FLOAT_EQ(50, result_point2.point.y);
  EXPECT_FLOAT_EQ(0, result_point2.point.z);
}

TEST(TestSuite, test_param_parse_poly)
{
  std::vector<geometry_msgs::Point> poly;
  EXPECT_FALSE(ParsePolygonParam("", poly));
  EXPECT_TRUE(ParsePolygonParam("(1, 2)", poly));
  EXPECT_FLOAT_EQ(poly[0].x, 1);
  EXPECT_FLOAT_EQ(poly[0].y, 2);
  EXPECT_FLOAT_EQ(poly[0].z, 0);
  EXPECT_FALSE(ParsePolygonParam("( 0 ,)", poly));
  EXPECT_FALSE(ParsePolygonParam("(  ,)", poly));
}

TEST(TestSuite, test_param_parse_filename)
{
  std::vector<std::string> names;
  EXPECT_FALSE(ParseFilenamesParam("", names));
  EXPECT_FALSE(ParseFilenamesParam("hello,", names));
  EXPECT_TRUE(ParseFilenamesParam("hello a ,there", names));
  EXPECT_STREQ(names[0].c_str(), "hello a ");
  EXPECT_STREQ(names[1].c_str(), "there");
}


// Run all the tests that were declared with TEST()
//int main(int argc, char **argv){
//  testing::InitGoogleTest(&argc, argv);
//  ros::init(argc, argv, "tester");
//  ros::NodeHandle nh;
//  return RUN_ALL_TESTS();
//}
