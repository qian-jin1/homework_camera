#include "kalman.hpp"
#include <cmath>

// 构造函数：初始化矩阵
ArmorKalmanFilter::ArmorKalmanFilter() : is_initialized(false) {
    state = cv::Mat::zeros(6, 1, CV_64F);
    P = cv::Mat::eye(6, 6, CV_64F) * 1000;  
    F = cv::Mat::eye(6, 6, CV_64F);
    H = cv::Mat::zeros(3, 6, CV_64F);
    Q = cv::Mat::eye(6, 6, CV_64F);
    R = cv::Mat::eye(3, 3, CV_64F) * 10;   

    // 测量矩阵
    H.at<double>(0, 0) = 1;
    H.at<double>(1, 1) = 1;
    H.at<double>(2, 2) = 1;

    // 过程噪声
    Q.at<double>(3, 3) = 0.1;
    Q.at<double>(4, 4) = 0.1;
    Q.at<double>(5, 5) = 0.1;
}

// 初始化状态
void ArmorKalmanFilter::initialize(const cv::Mat& measure_pos) {
    if (measure_pos.rows != 3 || measure_pos.cols != 1) {
        throw std::runtime_error("测量位置必须是3x1矩阵");
    }
    state.at<double>(0) = measure_pos.at<double>(0);  
    state.at<double>(1) = measure_pos.at<double>(1);  
    state.at<double>(2) = measure_pos.at<double>(2);  
    state.at<double>(3) = 0; 
    state.at<double>(4) = 0;  
    state.at<double>(5) = 0;  

    last_time = std::chrono::steady_clock::now();
    is_initialized = true;
}

// 预测步骤
void ArmorKalmanFilter::predict() {
    if (!is_initialized) return;

    // 计算Δt
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_time).count();
    last_time = current_time;

    // 更新状态转移矩阵
    F.at<double>(0, 3) = dt;
    F.at<double>(1, 4) = dt;
    F.at<double>(2, 5) = dt;

    // 预测状态和协方差
    state = F * state;
    P = F * P * F.t() + Q;
}

// 更新步骤
void ArmorKalmanFilter::update(const cv::Mat& measure_pos) {
    if (!is_initialized) {
        initialize(measure_pos);
        return;
    }

    cv::Mat y = measure_pos - H * state;  // 残差
    cv::Mat S = H * P * H.t() + R;        // 残差协方差
    cv::Mat K = P * H.t() * S.inv();      // 卡尔曼增益

    // 更新状态和协方差
    state = state + K * y;
    cv::Mat I = cv::Mat::eye(6, 6, CV_64F);
    P = (I - K * H) * P;
}

// 获取预测位置
cv::Mat ArmorKalmanFilter::get_predicted_pos() const {
    cv::Mat pos = cv::Mat::zeros(3, 1, CV_64F);
    pos.at<double>(0) = state.at<double>(0);
    pos.at<double>(1) = state.at<double>(1);
    pos.at<double>(2) = state.at<double>(2);
    return pos;
}

// 获取预测速度
cv::Mat ArmorKalmanFilter::get_predicted_vel() const {
    cv::Mat vel = cv::Mat::zeros(3, 1, CV_64F);
    vel.at<double>(0) = state.at<double>(3);
    vel.at<double>(1) = state.at<double>(4);
    vel.at<double>(2) = state.at<double>(5);
    return vel;
}
