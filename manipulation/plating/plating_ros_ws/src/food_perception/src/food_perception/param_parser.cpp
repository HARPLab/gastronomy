#include "food_perception/param_parser.h"

bool ParseFilenamesParam(std::string names_raw, std::vector<std::string> &names)
{
  //https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
  boost::split(names, names_raw, [](char c){return c == ',';});
  for (std::vector<std::string>::iterator itr = names.begin(); itr != names.end(); itr++)
  {
    if (itr->size() == 0)
    {
      return false;
    }
  }
  return true;
}

// params:
// roi_polygon_raw_str: a string of the form '(x0, y0), (x1, y1), (x2, y2), ...' 
// poly_of_interest: an out parameter that's filled with a vector of geometry_msgs::Point representation of the raw_str input
// returns:
// a bool indicating successful filling out of poly of interst
bool ParsePolygonParam(std::string roi_polygon_raw_str, std::vector<geometry_msgs::Point> &poly_of_interest)
{
  if (roi_polygon_raw_str.size() == 0)
  {
    return false;
  }

  std::vector<std::string> polygon_coords_str;

  //https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
  boost::split(polygon_coords_str, roi_polygon_raw_str, [](char c){return c == ' ' || c == '(' || c == ',' || c == ')';});

  
  std::vector<std::string>::iterator itr = polygon_coords_str.begin();
  // move to the first non-empty character string
  while (itr->size() == 0)
  {
    itr++;
    if (itr == polygon_coords_str.end())
    {
      return false;
    }
  }

  while (itr != polygon_coords_str.end())
  {
    double x = atof(itr->c_str());
    bool has_moved = false;
    while (itr->size() == 0 || !has_moved)
    {
      itr++;
      has_moved = true;
      if (itr == polygon_coords_str.end())
      {
        return false;
      }
    }
    double y = atof(itr->c_str());

    has_moved = false;
    while (!has_moved || (itr != polygon_coords_str.end() && itr->size() == 0))
    {
      has_moved = true;
      itr++;
    }
    // there's no nice constructor like there is in python https://github.com/ros/ros_comm/issues/148
    geometry_msgs::Point pt;
    pt.x = x;
    pt.y = y;
    pt.z = 0.0;
    poly_of_interest.push_back(pt);
  }
  return true;
}
