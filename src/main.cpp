#include "tasks/detector.hpp"
#include "tools/img_tools.hpp"
#include "io/camera.hpp"       
#include "kalman.hpp"   
#include "fmt/core.h"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <map>

// 取消注释下行启用相机模式，注释则启用视频模式
//#define USE_CAMERA  

#define TEST_CAMERA_PITCH 0    
#define PRINT_PREDICTION true  


std::map<int, ArmorKalmanFilter> armor_kalman_filters;
int next_armor_id = 1;

// 装甲板ID
int get_armor_id(const cv::Mat& world_pos) {
    const double MAX_DISTANCE = 0.5;  
    double min_dist = MAX_DISTANCE;
    int matched_id = -1;

    for (const auto& pair : armor_kalman_filters) {
        int id = pair.first;
        const auto& kf = pair.second;
        cv::Mat pred_pos = kf.get_predicted_pos();
        
        // 计算欧式距离
        double dx = world_pos.at<double>(0) - pred_pos.at<double>(0);
        double dy = world_pos.at<double>(1) - pred_pos.at<double>(1);
        double dz = world_pos.at<double>(2) - pred_pos.at<double>(2);
        double dist = sqrt(dx*dx + dy*dy + dz*dz);

        if (dist < min_dist) {
            min_dist = dist;
            matched_id = id;
        }
    }

    if (matched_id == -1) {
        matched_id = next_armor_id++;
        armor_kalman_filters[matched_id] = ArmorKalmanFilter();
    }
    return matched_id;
}

// 工具函数：角度转换
static double deg2rad(double deg) { return deg * CV_PI / 180.0; }
static double rad2deg(double rad) { return rad * 180 / CV_PI; }

// 相机Pitch角旋转矩阵
static cv::Mat get_rotation_matrix_x(double pitch_deg) {
    double theta = deg2rad(pitch_deg);
    return (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(theta), -sin(theta),
        0, sin(theta), cos(theta)
    );
}

// 相机坐标转世界坐标
static cv::Mat camera_to_world(const cv::Mat& camera_p, double camera_pitch_deg) {
    cv::Mat R = get_rotation_matrix_x(camera_pitch_deg);
    return R.t() * camera_p;  
}

// 弹道计算：弹丸下落距离
static double calc_bullet_drop(double& flight_time, double x, double theta, double v0 = 28.5) {
    double v_z0 = v0 * cos(theta);
    flight_time = (exp(0.01 * x) - 1) / (0.01 * v_z0);
    return v0 * sin(theta) * flight_time - 0.5 * 9.8 * flight_time * flight_time;
}

// 求解瞄准Pitch角
static double calc_pitch_angle(double x, double y) {
    double theta = 0.0, aim_y = y;
    for (int i = 0; i < 50; ++i) {
        double flight_time;
        double drop = calc_bullet_drop(flight_time, x, theta);
        double error = aim_y - drop;
        if (fabs(error) < 0.001) break;
        aim_y += error * 0.5;
        theta = atan2(aim_y, x);
    }
    return theta;
}

// 求解瞄准Yaw角
static double calc_yaw_angle(double x, double z) {
    return atan2(x, z);
}

// 安全绘制文字
static void safe_draw_text(cv::Mat& img, const std::string& text, cv::Point2f pos,
                          double font_scale = 0.55, cv::Scalar color = cv::Scalar(255,255,255), int thickness = 1) {
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    if (pos.x + text_size.width > img.cols) pos.x = img.cols - text_size.width - 10;
    pos.y = std::max(static_cast<float>(baseline + 10), std::min(pos.y, static_cast<float>(img.rows - 10)));
    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

// 装甲板处理逻辑
void process_armors(const std::vector<auto_aim::Armor>& armors, cv::Mat& img,
                   const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs,
                   const std::vector<cv::Point3f>& object_points) {
    for (const auto& armor : armors) {
        // 提取图像坐标点
        std::vector<cv::Point2f> img_pts = {
            armor.left.top, armor.right.top,
            armor.right.bottom, armor.left.bottom
        };

        // PnP解算相机坐标
        cv::Mat rvec, tvec_cam;
        cv::solvePnP(object_points, img_pts, camera_matrix, distort_coeffs, rvec, tvec_cam);

        // 转换为世界坐标
        cv::Mat tvec_world = camera_to_world(tvec_cam, TEST_CAMERA_PITCH);

        // 卡尔曼滤波
        int armor_id = get_armor_id(tvec_world);
        auto& kf = armor_kalman_filters[armor_id];
        kf.predict();
        kf.update(tvec_world);

        // 预测结果
        cv::Mat pred_pos = kf.get_predicted_pos();
        cv::Mat pred_vel = kf.get_predicted_vel();

        // 6D位姿计算
        cv::Mat rmat;
        cv::Rodrigues(rvec, rmat);
        double yaw_deg = rad2deg(atan2(rmat.at<double>(1,0), rmat.at<double>(0,0)));
        double pitch_deg = rad2deg(atan2(-rmat.at<double>(2,0), 
                                sqrt(pow(rmat.at<double>(2,1),2) + pow(rmat.at<double>(2,2),2))));
        double roll_deg = rad2deg(atan2(rmat.at<double>(2,1), rmat.at<double>(2,2)));

        // 瞄准角计算
        double aim_yaw_deg = rad2deg(calc_yaw_angle(tvec_world.at<double>(0), tvec_world.at<double>(2)));
        double aim_pitch_deg = rad2deg(calc_pitch_angle(tvec_world.at<double>(2), tvec_world.at<double>(1)));

        // 打印信息
        if (PRINT_PREDICTION) {
            fmt::print("【装甲板 #{}】\n", armor_id);
            fmt::print("  测量位置:x={:.4f}, y={:.4f}, z={:.4f} 米\n",
                      tvec_world.at<double>(0), tvec_world.at<double>(1), tvec_world.at<double>(2));
            fmt::print("  预测位置:x={:.4f}, y={:.4f}, z={:.4f} 米\n",
                      pred_pos.at<double>(0), pred_pos.at<double>(1), pred_pos.at<double>(2));
            fmt::print("  预测速度:vx={:.4f}, vy={:.4f}, vz={:.4f} 米/秒\n",
                      pred_vel.at<double>(0), pred_vel.at<double>(1), pred_vel.at<double>(2));
            fmt::print("  6D位姿:yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°\n",
                      yaw_deg, pitch_deg, roll_deg);
            fmt::print("  瞄准角:yaw={:.2f}°, pitch={:.2f}°\n",
                      aim_yaw_deg, aim_pitch_deg);
        }

        // 图像绘制
        cv::Point2f top_mid((armor.points[0].x + armor.points[1].x)/2, armor.points[0].y - 20);
        
// 测量位置
safe_draw_text(img, fmt::format("Meas: [{:.2f},{:.2f},{:.2f}]",
              tvec_world.at<double>(0),
              tvec_world.at<double>(1),
              tvec_world.at<double>(2)),
              cv::Point2f(top_mid.x, top_mid.y + 20), 0.55, cv::Scalar(0, 255, 0), 1);

// 预测位置
    safe_draw_text(img, fmt::format("Pred: [{:.2f},{:.2f},{:.2f}]",
              pred_pos.at<double>(0),
              pred_pos.at<double>(1),
              pred_pos.at<double>(2)),
              cv::Point2f(top_mid.x, top_mid.y + 40), 0.55, cv::Scalar(0, 165, 255), 1);

// 预测速度
    safe_draw_text(img, fmt::format("Vel: [{:.2f},{:.2f},{:.2f}]",
              pred_vel.at<double>(0),
              pred_vel.at<double>(1),
              pred_vel.at<double>(2)),
              cv::Point2f(top_mid.x, top_mid.y + 60), 0.55, cv::Scalar(128, 0, 128), 1);

    }
}

int main(int argc, char *argv[]) {
    // 相机内参和畸变系数
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
        3459.981717, 0.000000, 1523.076759,
        0.000000, 3469.145301, 1019.198285,
        0.000000, 0.000000, 1.000000
    );
    cv::Mat distort_coeffs = (cv::Mat_<double>(5,1) <<
        -0.091955, 0.395148, 0.003058, -0.000456, 0.000000
    );

    // 装甲板局部3D坐标（米）
    const double LIGHTBAR_LENGTH = 0.056;
    const double ARMOR_WIDTH = 0.135;
    std::vector<cv::Point3f> object_points = {
        {-ARMOR_WIDTH/2, -LIGHTBAR_LENGTH/2, 0},
        {ARMOR_WIDTH/2,  -LIGHTBAR_LENGTH/2, 0},
        {ARMOR_WIDTH/2,  LIGHTBAR_LENGTH/2,  0},
        {-ARMOR_WIDTH/2, LIGHTBAR_LENGTH/2,  0}
    };

    // 初始化检测器和图像
    auto_aim::Detector detector;
    cv::Mat img;  

    // 系统信息
    fmt::print("=== 装甲板卡尔曼滤波预测系统 ===\n");
    fmt::print("当前相机Pitch角：{}°\n", TEST_CAMERA_PITCH);

#ifdef USE_CAMERA
    // 相机模式
    fmt::print("运行模式：相机（按 'q' 退出）\n");
    try {
        io::Camera camera(50, 0);  
        while (true) {
            camera.read(img);
            if (img.empty()) {
                fmt::print("相机图像读取失败！\n");
                break;
            }
            cv::resize(img, img, cv::Size(), 0.6, 0.6);

            // 检测并处理装甲板
            auto armors = detector.detect(img);
            fmt::print("\n检测到 {} 个装甲板\n", armors.size());
            process_armors(armors, img, camera_matrix, distort_coeffs, object_points);

            // 显示图像
            cv::imshow("装甲板跟踪", img);
            if (cv::waitKey(1) == 'q') break;
        }
        fmt::print("相机模式退出\n");
    } catch (const std::exception& e) {
        fmt::print("相机错误：{}\n", e.what());
        return -1;
    }
#else
    // 视频模式
    fmt::print("运行模式：视频（按 'q' 退出）\n");
    cv::VideoCapture cap("../测试视频2.webm");  
    if (!cap.isOpened()) {
        fmt::print("无法打开视频！请检查路径\n");
        return -1;
    }

    const int WAIT_MS = 33; 
   
while (true) {
    cap >> img;
    if (img.empty()) {
        fmt::print("视频播放结束\n");
        break;
    }
    cv::resize(img, img, cv::Size(), 0.6, 0.6);

    // 检测装甲板
    auto armors_list = detector.detect(img);  
    std::vector<auto_aim::Armor> armors(armors_list.begin(), armors_list.end());  

    fmt::print("\n检测到 {} 个装甲板\n", armors.size());
    process_armors(armors, img, camera_matrix, distort_coeffs, object_points);  

    cv::imshow("装甲板跟踪", img);
    if (cv::waitKey(WAIT_MS) == 'q') break;
}


    cap.release();
    fmt::print("视频模式退出\n");
#endif

    cv::destroyAllWindows();
    return 0;
}
