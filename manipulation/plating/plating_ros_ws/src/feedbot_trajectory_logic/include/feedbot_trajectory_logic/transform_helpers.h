#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <math.h>

double quat_distance(const Eigen::Quaterniond &cur_quat, const Eigen::Quaterniond &end_quat);
double distance(const Eigen::Translation3d &cur_trans, const Eigen::Translation3d &end_trans);
Eigen::Matrix<double,6,6> compute_jacob_from_rect_to_cyl(const Eigen::Vector3d &rect_coord);
Eigen::Vector3d get_cylindrical_point_translation(const Eigen::Vector3d &cur_loc, const Eigen::Vector3d &end_loc);
Eigen::Vector3d convert_rect_to_cyl(const Eigen::Vector3d &cur_loc);

