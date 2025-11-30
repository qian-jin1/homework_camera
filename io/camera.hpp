#ifndef IO_CAMERA__HPP
#define IO_CAMERA__HPP

#include <opencv2/opencv.hpp>

namespace io
{
    class Camera
    {
    public:
        // 构造函数：曝光时间（毫秒，-1为自动），相机索引
        Camera(int exposure = -1, int camera_index = 0);
        ~Camera();
        // 读取一帧图像到cv::Mat
        void read(cv::Mat &img);

    private:
        int exposure_ms_;         // 曝光时间（毫秒）
        void* handle_;            // 海康相机句柄
        //unsigned char* rgb_buf_;  // BGR格式图像缓存
        int width_;               // 图像宽度
        int height_;              // 图像高度
        bool is_opened_;          // 是否初始化成功
    };
} 

#endif
