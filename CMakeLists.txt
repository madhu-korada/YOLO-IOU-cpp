Cmake_minimum_required(VERSION 2.8)
project( yolo_iou )
set (CMAKE_CXX_STANDARD 11)

# find_package(OpenCV REQUIRED)
find_package(OpenCV 4.1.1 REQUIRED PATHS "/home/madhu/installation/OpenCV-master/lib/cmake/opencv4")

include_directories(${OpenCV_INCLUDE_DIRS}) # to add include directories
add_executable( yolo_iou object_detection_yolo.cpp ) 
target_link_libraries(yolo_iou ${OpenCV_LIBS}) # add opencv libs to out execuatable

