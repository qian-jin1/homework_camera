#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <chrono>

// 卡尔曼滤波器
class ArmorKalmanFilter {
public:
    cv::Mat state;  // 状态向量
    cv::Mat P;      // 状态协方差矩阵
    cv::Mat F;      // 状态转移矩阵
    cv::Mat H;      // 测量矩阵
    cv::Mat Q;      // 过程噪声协方差
    cv::Mat R;      // 测量噪声协方差
    std::chrono::steady_clock::time_point last_time;
    bool is_initialized;

    ArmorKalmanFilter();  // 构造函数
    void initialize(const cv::Mat& measure_pos);  // 初始化状态
    void predict();  // 预测步骤
    void update(const cv::Mat& measure_pos);  // 更新步骤
    cv::Mat get_predicted_pos() const;  // 获取预测位置
    cv::Mat get_predicted_vel() const;  // 获取预测速度
};

#endif // KALMAN_FILTER_HPP
