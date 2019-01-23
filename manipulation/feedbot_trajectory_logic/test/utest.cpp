// see http://wiki.ros.org/gtest
// Bring in my package's API, which is what I'm testing
#include "feedbot_trajectory_logic/transform_helpers.h"
// Bring in gtest
#include <gtest/gtest.h>

// Declare a test
TEST(TestSuite, distance_function)
{
  Eigen::Vector3d a(0,0,1), b(1,-1,2);
  EXPECT_FLOAT_EQ(distance(Eigen::Translation3d(a), Eigen::Translation3d(b)), std::sqrt(3));
  EXPECT_FLOAT_EQ(distance(Eigen::Translation3d(a), Eigen::Translation3d(a)), 0);
  EXPECT_FLOAT_EQ(distance(Eigen::Translation3d(b), Eigen::Translation3d(a)), std::sqrt(3));
}

TEST(TestSuite, quat_distance_function)
{
  Eigen::Vector3d axis_a(0,0,1), axis_b(1,0,0);
  Eigen::Quaternion<double> q_0(Eigen::AngleAxis<double>(0, axis_a)), 
                     q_1(Eigen::AngleAxis<double>(M_PI,axis_a)), 
                     q_2(Eigen::AngleAxis<double>(M_PI/2, axis_b)),
                     q_3(Eigen::AngleAxis<double>(M_PI/2, axis_a));
  EXPECT_FLOAT_EQ(quat_distance(q_0, q_0), 0);
  EXPECT_FLOAT_EQ(quat_distance(q_0, q_1), M_PI);
  EXPECT_FLOAT_EQ(quat_distance(q_0, q_2), M_PI/2);
  EXPECT_FLOAT_EQ(quat_distance(q_1, q_2), M_PI);
  EXPECT_FLOAT_EQ(quat_distance(q_1, q_3), M_PI/2);
}

TEST(TestSuite, jacob_from_cyl_to_rect)
{
  Eigen::Vector3d point_a(0,1,0), point_b(1,0,0);
  Eigen::Matrix<double,6,6> expected_a, expected_b;
  expected_a << 0, 1, 0,0,0,0,
                -1, 0, 0,0,0,0,
                0, 0, 1,0,0,0,
                0,0,0,1,0,0,
                0,0,0,0,1,0,
                0,0,0,0,0,1; 
  expected_b << 1, 0, 0,0,0,0,
                0, 1, 0,0,0,0,
                0, 0, 1,0,0,0,
                0,0,0,1,0,0,
                0,0,0,0,1,0,
                0,0,0,0,0,1; 
  ASSERT_TRUE(compute_jacob_from_rect_to_cyl(point_a).isApprox(expected_a));
  ASSERT_TRUE(compute_jacob_from_rect_to_cyl(point_b).isApprox(expected_b));
}

TEST(TestSuite, get_cylindrical_point_translation_function)
{
  Eigen::Vector3d point_a(0,1,0), point_b(1,0,0),
                  point_c(0,3,3), point_d(1,0,4),
                  point_e(1,-1,-2), point_f(1,1,2);
  Eigen::Vector3d expected_1(0,-M_PI/2.0,0), 
                  expected_2(-2,-M_PI/2.0,1),
                  expected_3(0, M_PI/2.0, 4);
  ASSERT_TRUE(get_cylindrical_point_translation(point_a, point_b).isApprox(expected_1));
  ASSERT_TRUE(get_cylindrical_point_translation(point_c, point_d).isApprox(expected_2));
  ASSERT_TRUE(get_cylindrical_point_translation(point_e, point_f).isApprox(expected_3));
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  //ros::init(argc, argv, "tester");
  //ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
