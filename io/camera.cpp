#include "camera.hpp"
extern "C" {
#include "MvCameraControl.h"  
}
#include <stdexcept>
#include <cstring>
#include <fmt/core.h>
#include <iostream>
namespace io
{
    // 构造函数：初始化相机
    Camera::Camera(int exposure, int camera_index)
        : exposure_ms_(exposure), handle_(nullptr),
          width_(0), height_(0), is_opened_(false)
    {
        //  初始化SDK
        if (MV_CC_Initialize() != MV_OK) {
            throw std::runtime_error("海康SDK初始化失败");
        }

        // 枚举设备
        MV_CC_DEVICE_INFO_LIST dev_list;
        memset(&dev_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        int ret = MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, &dev_list);
        if (ret != MV_OK || dev_list.nDeviceNum <= camera_index) {
            MV_CC_Finalize();  // 反初始化SDK
            throw std::runtime_error("未找到指定相机设备");
        }

        //  创建设备句柄
        ret = MV_CC_CreateHandle(&handle_, dev_list.pDeviceInfo[camera_index]);
        if (ret != MV_OK) {
            MV_CC_Finalize();
            throw std::runtime_error("创建相机句柄失败");
        }

        // 打开设备
        ret = MV_CC_OpenDevice(handle_);
        if (ret != MV_OK) {
            MV_CC_DestroyHandle(handle_);
            MV_CC_Finalize();
            throw std::runtime_error("打开相机设备失败");
        }

        //  获取图像宽高
        MVCC_INTVALUE int_val;
        if (MV_CC_GetIntValue(handle_, "Width", &int_val) == MV_OK) {
            width_ = int_val.nCurValue;
        } else {
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            MV_CC_Finalize();
            throw std::runtime_error("获取图像宽度失败");
        }
        if (MV_CC_GetIntValue(handle_, "Height", &int_val) == MV_OK) {
            height_ = int_val.nCurValue;
        } else {
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            MV_CC_Finalize();
            throw std::runtime_error("获取图像高度失败");
        }

        //  配置曝光
        if (exposure >= 0) {
            // 关闭自动曝光
            ret = MV_CC_SetEnumValue(handle_, "ExposureAuto", 0);
            if (ret == MV_OK) {
                // 设置手动曝光
                ret = MV_CC_SetExposureTime(handle_, exposure * 1000);
                if (ret != MV_OK) {
                    fmt::print("警告：手动曝光设置失败，启用自动曝光\n");
                    MV_CC_SetEnumValue(handle_, "ExposureAuto", 1);
                }
            } else {
                fmt::print("警告：关闭自动曝光失败，使用自动曝光\n");
            }
        } else {
            // 启用自动曝光
            MV_CC_SetEnumValue(handle_, "ExposureAuto", 1);
        }

        

        //  开始取流
        ret = MV_CC_StartGrabbing(handle_);
        if (ret != MV_OK) {
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            MV_CC_Finalize();
            throw std::runtime_error("启动图像采集失败");
        }

        is_opened_ = true;
        fmt::print("海康相机初始化成功（宽：{}，高：{}，曝光：{}ms）\n",
                  width_, height_, exposure);
    }

    // 析构函数：释放资源
    Camera::~Camera()
    {
        if (is_opened_) {
            MV_CC_StopGrabbing(handle_);           
            MV_CC_CloseDevice(handle_);          
            MV_CC_DestroyHandle(handle_);
          
       
        }
        // 反初始化SDK
        MV_CC_Finalize();
    }

    // 读取图像
    void Camera::read(cv::Mat &img)
{
    if (!is_opened_) {
        throw std::runtime_error("相机未初始化，无法读取图像");
    }

    MV_FRAME_OUT frame_out;
    memset(&frame_out, 0, sizeof(MV_FRAME_OUT));
    int ret = MV_CC_GetImageBuffer(handle_, &frame_out, 3000);
    if (ret != MV_OK) {
        throw std::runtime_error("获取图像帧失败");
    }

    // 动态分配单通道缓存
    int buf_size = frame_out.stFrameInfo.nWidth * frame_out.stFrameInfo.nHeight * 1;
    unsigned char* temp_buf = new (std::nothrow) unsigned char[buf_size];
    if (!temp_buf) {
        MV_CC_FreeImageBuffer(handle_, &frame_out);
        throw std::runtime_error("临时缓存分配失败");
    }

    // 复制原始Bayer数据
    memcpy(temp_buf, frame_out.pBufAddr, buf_size);

  
    cv::Mat bayer_img(frame_out.stFrameInfo.nHeight, frame_out.stFrameInfo.nWidth, CV_8UC1, temp_buf);
    cv::cvtColor(bayer_img, img, cv::COLOR_BayerRG2BGR);  

    // 释放资源
    delete[] temp_buf;
    MV_CC_FreeImageBuffer(handle_, &frame_out);
}


} 
