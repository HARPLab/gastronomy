#include <feedbot_trajectory_logic/transform_helpers.h>
#include <iostream>

//Helper functions 
double quat_distance(const Eigen::Quaterniond &cur_quat, const Eigen::Quaterniond &end_quat)
{
  Eigen::Quaterniond diff =  cur_quat.inverse() * end_quat;
  Eigen::AngleAxisd rot_axis_angle(diff);
  double rot_angle = rot_axis_angle.angle();
  return rot_angle; 
}

double distance(const Eigen::Translation3d &cur_trans, const Eigen::Translation3d &end_trans)
{
  Eigen::Translation3d diff = cur_trans.inverse() * end_trans;
  return std::sqrt(std::pow(diff.x(), 2) + std::pow(diff.y(), 2) + std::pow(diff.z(),2));
}

// Compute the jacobian of the transformation between rect and cylindrical
// That is,  we define r = sqrt(x^2 + y^2)
// theta = arctan2(y,x)
// z = z
// and we compute
// dr/dx, dr/dy, dr/dz
// dtheta/dx, dtheta/dy
// dz/dx, dz/dy, dz/dz
// for the input vector X,Y,Z
Eigen::Matrix<double,6,6> compute_jacob_from_rect_to_cyl(const Eigen::Vector3d &rect_coord)
{
  double x = rect_coord[0];
  double y = rect_coord[1];
  double z = rect_coord[2]; 
  //validated with http://www.wolframalpha.com/input/?i=d%2Fdx+arctan(y%2Fx)
  double xyr = std::pow(std::pow(x,2.0) + std::pow(y, 2.0), 0.5);
  //std::cout << "we have x,y,z" << x << "," <<y<<","<<z<<","<< " and xyr " << xyr << std::endl;
  // yeah, yeah, yeah, this will cause massive problems if we are ever at exactly pi/2, pi, 0, etc. But I think it's worth it? We'll see!
  Eigen::Matrix<double,6,6> rect_to_cyl_jacob;
  rect_to_cyl_jacob  << x/xyr, y/xyr, 0,                           0,0,0,
               -y /std::pow(xyr, 2.0), x/ std::pow(xyr, 2.0), 0,   0,0,0,
               0,0,1,                                              0,0,0,
               0,0,0,1,0,0,
               0,0,0,0,1,0,
               0,0,0,0,0,1;
  return rect_to_cyl_jacob;
}

// get the difference between two cartesian points, with the result given in cylindrical coordinates
Eigen::Vector3d get_cylindrical_point_translation(const Eigen::Vector3d &cur_loc, const Eigen::Vector3d &end_loc)
{
  Eigen::Vector3d cur_cyl = convert_rect_to_cyl(cur_loc);  
  Eigen::Vector3d end_cyl = convert_rect_to_cyl(end_loc);  
  double dr = end_cyl[0] - cur_cyl[0];
  double dtheta = end_cyl[1] - cur_cyl[1];
  double dz = end_cyl[2] - cur_cyl[2];
  if (dtheta > M_PI)
  {
    dtheta = dtheta - 2 * M_PI;
  }
  if (dtheta < - M_PI)
  {
    dtheta = dtheta + 2 * M_PI;
  }
  return Eigen::Vector3d(dr, dtheta, dz);
}

Eigen::Vector3d
convert_rect_to_cyl(const Eigen::Vector3d &cur_loc)
{
  double x = cur_loc[0];
  double y = cur_loc[1];
  double z = cur_loc[2];
  double r = std::pow(std::pow(x,2.0) + std::pow(y,2.0), 0.5);
  double theta = atan2(y, x);
  Eigen::Vector3d cyl_coord;
  cyl_coord << r, theta, z;
  return cyl_coord;
}
