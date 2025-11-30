cmake_minimum_required(VERSION 3.16.3)

project(homework)

set(CMAKE_CXX_STANDARD 17)
set(MVS_ROOT /opt/MVS)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${MVS_ROOT}/include  
    ${MVS_ROOT}/include/MvCameraControl  
)
link_directories(${MVS_ROOT}/lib/64)

add_executable(main src/main.cpp tasks/detector.cpp io/camera.cpp src/kalman.cpp)
target_link_libraries(main ${OpenCV_LIBS} fmt::fmt MvCameraControl  pthread dl)