#include "camera.hpp"
#include <stdexcept>
#include <iostream>

namespace io
{
   
    Camera::Camera(int exposure, int camera_index) 
        : exposure_ms_(exposure), is_opened_(false)
    {
        // 打开相机
        cap_.open(camera_index);
        if (!cap_.isOpened())
        {
            throw std::runtime_error("无法打开相机！请检查相机连接或索引是否正确。");
        }

        // 设置曝光（-1表示使用自动曝光）
        if (exposure >= 0)
        {
            
            if (!cap_.set(cv::CAP_PROP_EXPOSURE, exposure))
            {
                std::cerr << "警告：相机可能不支持手动设置曝光，将使用自动曝光。" << std::endl;
            }
        }
        else
        {
            // 启用自动曝光
            cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0); // 1.0表示自动曝光模式
        }

        is_opened_ = true;
    }

    
    Camera::~Camera()
    {
        if (is_opened_)
        {
            cap_.release(); // 关闭相机
        }
    }

    // 读取一帧图像
    void Camera::read(cv::Mat &img)
    {
        if (!is_opened_)
        {
            throw std::runtime_error("相机未初始化，请先确保构造函数成功执行。");
        }

        // 读取图像
        cap_.read(img);
        if (img.empty())
        {
            throw std::runtime_error("读取图像失败！可能是相机断开或视频结束。");
        }
    }
} // namespace io
