#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class FoodPixelIdentifier
{
  public:
    bool GetFoodPixelCenter(const cv::Mat &image, cv::Point2d &pixel);
};

