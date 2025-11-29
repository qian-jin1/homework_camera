#ifndef IO_CAMERA__HPP
#define IO_CAMERA__HPP

#include <opencv2/opencv.hpp>

namespace io
{
    class Camera
    {
    public:
        // 构造函数：exposure为曝光时间（毫秒，-1表示自动曝光）
        Camera(int exposure = -1, int camera_index = 0);
        ~Camera();
        // 读取一帧图像到img
        void read(cv::Mat &img);

    private:
        int exposure_ms_;          
        cv::VideoCapture cap_;     
        bool is_opened_;           
    };
} // namespace io

#endif
