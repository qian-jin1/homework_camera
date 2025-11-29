#include "tasks/detector.hpp"
#include "tools/img_tools.hpp"
#include "fmt/core.h"
#include <cmath>

//#define USE_CAMERA  // 取消注释此行切换到相机模式，注释此行切换到视频模式

#ifdef USE_CAMERA
#include "io/camera.hpp"  
#endif


// 相机内参
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    3459.981717, 0.000000, 1523.076759,
    0.000000, 3469.145301, 1019.198285,
    0.000000, 0.000000, 1.000000
);

// 相机畸变系数
cv::Mat distort_coeffs = (cv::Mat_<double>(5, 1) <<
    -0.091955, 0.395148, 0.003058, -0.000456, 0.000000
);

// 装甲板尺寸参数
static const double LIGHTBAR_LENGTH = 0.056;
static const double ARMOR_WIDTH = 0.135;

// 装甲板局部坐标
static const std::vector<cv::Point3f> object_points {
    {-ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2, 0 },
    {ARMOR_WIDTH / 2,  -LIGHTBAR_LENGTH / 2, 0 },
    {ARMOR_WIDTH / 2,  LIGHTBAR_LENGTH / 2,  0 },
    {-ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2,  0 }
};

// 弹丸与弹道参数
static const double BULLET_SPEED = 28.5;    // 弹速（27-30m/s）
static const double GRAVITY = 9.8;          // 重力加速度
static const double K1 = 0.01;              // 空气阻力系数
static const int MAX_ITERATE = 50;          // 迭代最大次数
static const double PRECISION = 0.001;      // 迭代精度（米）
static const double ITERATE_SCALE = 0.5;    // 迭代缩放因子

// 弹道结构体（存储飞行时间）
struct SolveTrajectory {
    double flight_time;  
};


 //角度转弧度

static double deg2rad(double deg) {
    return deg * CV_PI / 180.0;
}


 // 计算弹丸垂直落点

static double calc_bullet_drop(SolveTrajectory& traj, double x, double theta) {
    double v_z0 = BULLET_SPEED * cos(theta);  // 水平初速度分量
    traj.flight_time = (exp(K1 * x) - 1) / (K1 * v_z0);  // 飞行时间（空气阻力模型）
    double v_y0 = BULLET_SPEED * sin(theta);  // 垂直初速度分量
    return v_y0 * traj.flight_time - 0.5 * GRAVITY * pow(traj.flight_time, 2);  // 垂直落点
}


  //迭代求解命中目标的pitch角（垂直瞄准角）
 
static double calc_pitch_angle(SolveTrajectory& traj, double x, double y) {
    double theta = 0.0;  // 初始仰角
    double aim_y = y;    // 迭代瞄准垂直位置

    for (int i = 0; i < MAX_ITERATE; ++i) {
        double drop_y = calc_bullet_drop(traj, x, theta);
        double error = aim_y - drop_y;  

        if (fabs(error) < PRECISION) break;  

        aim_y += error * ITERATE_SCALE;      
        theta = atan2(aim_y, x);             
    }
    return theta;
}


 //计算yaw角（水平瞄准角）

static double calc_yaw_angle(double x, double z) {
    return atan2(x, z);  
}


int main(int argc, char *argv[]) {
   
    auto_aim::Detector detector;
    cv::Mat img;  
    SolveTrajectory traj;  // 弹道计算实例

#ifdef USE_CAMERA
    // -------------------------- 相机模式 --------------------------
    fmt::print("当前模式：相机（按 'q' 退出）\n");
    try {
        // 初始化相机：参数1=曝光时间（-1=自动曝光），参数2=相机索引（默认0）
        io::Camera camera(-1, 0);

        // 主循环：读取相机帧并处理
        while (true) {
            camera.read(img);  
            if (img.empty()) {
                fmt::print("相机图像读取失败！\n");
                break;
            }
#else
    // -------------------------- 视频模式 --------------------------
    fmt::print("当前模式：测试视频（按 'q' 退出）\n");
    // 初始化视频读取
    cv::VideoCapture cap("../测试视频2.webm", cv::CAP_ANY);
    if (!cap.isOpened()) {
        fmt::print("Error: 无法打开测试视频！请检查视频路径是否正确\n");
        return -1;
    }

    // 主循环：读取视频帧并处理
    while (true) {
        cap >> img;  
        if (img.empty()) break;  
#endif

        // -------------------------- 两种模式共用 --------------------------
        // 1. 装甲板识别
        auto armors = detector.detect(img);
        
        // 2. 遍历所有装甲板并处理
        int armor_index = 0;
        for (const auto& armor : armors) {
            // 绘制装甲板框
            tools::draw_points(img, armor.points);

            // 提取像素坐标（PnP解算用）
            std::vector<cv::Point2f> img_points{
                armor.left.top, armor.right.top,
                armor.right.bottom, armor.left.bottom
            };

            // PnP解算位姿
            cv::Mat rvec, tvec;
            cv::solvePnP(object_points, img_points, camera_matrix, distort_coeffs, rvec, tvec);

            // 相机系坐标
            double target_x = tvec.at<double>(0);
            double target_y = tvec.at<double>(1);
            double target_z = tvec.at<double>(2);

            // 计算瞄准角
            double yaw = 0.0, pitch = 0.0;
            SolveTrajectory traj;
            if (target_z > 0.5) {  // 忽略过近距离目标
                yaw = calc_yaw_angle(target_x, target_z);
                pitch = calc_pitch_angle(traj, target_z, target_y);
            }

            // 计算6D位姿（欧拉角）
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);  // 旋转向量转矩阵
            double euler_yaw, euler_pitch, euler_roll;
            euler_pitch = atan2(-rmat.at<double>(2, 0),
                              sqrt(pow(rmat.at<double>(2, 1), 2) + pow(rmat.at<double>(2, 2), 2)));
            if (std::abs(cos(euler_pitch)) > 1e-6) {
                euler_yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
                euler_roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));
            } else {
                euler_yaw = 0.0;
                euler_roll = atan2(-rmat.at<double>(0, 1), rmat.at<double>(1, 1));
            }

            // 计算文字显示位置（跟随装甲板顶部中点）
            cv::Point2f top_left = armor.points[0];   
            cv::Point2f top_right = armor.points[1];  
            cv::Point2f text_base_pos(
                (top_left.x + top_right.x) / 2,  
                top_left.y - 5                   
            );

        
            double font_scale = 0.7;
            cv::Scalar text_color = cv::Scalar(0, 255, 255);

            // 显示6D位姿（yaw/pitch/roll）
            tools::draw_text(img, fmt::format("{:.2f} {:.2f} {:.2f}",
                euler_yaw*180/CV_PI, euler_pitch*180/CV_PI, euler_roll*180/CV_PI),
                cv::Point2f(text_base_pos.x, text_base_pos.y + 20),  
                font_scale, text_color, 2);

            // 显示瞄准角（yaw/pitch）
            tools::draw_text(img, fmt::format("{:.2f} {:.2f}",
                yaw*180/CV_PI, pitch*180/CV_PI),
                cv::Point2f(text_base_pos.x, text_base_pos.y + 40),  
                font_scale, text_color, 2);

            armor_index++;
        }

        // 显示图像
        cv::Mat display_img;
        cv::resize(img, display_img, cv::Size(), 0.6, 0.6);
        cv::imshow("Armor Detection & Ballistic Calculation", display_img);

        // 按'q'退出
        if (cv::waitKey(20) & 0xFF == 'q') break;
    }

    // -------------------------- 资源释放--------------------------
#ifdef USE_CAMERA
    // 相机模式
    fmt::print("相机模式已退出\n");
#else
    // 视频模式：释放视频捕获资源
    cap.release();
    fmt::print("视频模式已退出\n");
#endif
    cv::destroyAllWindows();
    return 0;
#ifdef USE_CAMERA
    } catch (const std::exception& e) {
        // 捕获相机初始化异常（如未连接、权限不足）
        fmt::print("相机错误：{}\n", e.what());
        fmt::print("排查：1. 检查相机连接 2. 执行sudo usermod -aG video $USER 3. 重启电脑\n");
        return -1;
    }
#endif
}
