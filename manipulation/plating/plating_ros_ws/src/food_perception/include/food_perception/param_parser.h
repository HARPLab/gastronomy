#include <boost/algorithm/string.hpp>
#include <geometry_msgs/Point.h>
#include <vector>

// params:
// roi_polygon_raw_str: a string of the form '(x0, y0), (x1, y1), (x2, y2), ...' 
// poly_of_interest: an out parameter that's filled with a vector of geometry_msgs::Point representation of the raw_str input
// returns:
// a bool indicating successful filling out of poly of interst
bool ParsePolygonParam(std::string roi_polygon_raw_str, std::vector<geometry_msgs::Point> &poly_of_interest);

// params 
// names_raw: a string of the form '/path/to/file,/path/to/file/b,...' 
// names: an out parameter that's filled with a vector of file paths 
// returns:
// a bool indicating successful filling out names 
bool ParseFilenamesParam(std::string names_raw, std::vector<std::string> &names);
