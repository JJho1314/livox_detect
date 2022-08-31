cmake_minimum_required(VERSION 3.0)
project(livox_detect)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Debug")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        roslaunch
        autoware_msgs
        # cv_bridge    //开启这个会造成opencv的冲突
        pcl_conversions
        pcl_ros
        geometry_msgs
        laser_geometry
        sensor_msgs
        message_generation
        visualization_msgs
        # common_lib
        # object_builders_lib
        # grid_map_ros
        # grid_map_cv               //注释掉避免在xavier编译时与pcl1.8的kdtree_flannch
        # grid_map_msgs
        jsk_rviz_plugins
        jsk_recognition_msgs
)

generate_messages(
        DEPENDENCIES
        std_msgs
        geometry_msgs
        sensor_msgs
)

catkin_package(
        # INCLUDE_DIRS
        # LIBRARIES
        # CATKIN_DEPENDS roscpp rospy std_msgs
)

# set flags for CUDA availability
set(USE_CUDA ON)
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# find_package(autoware_build_flags REQUIRED)

# AW_CHECK_CUDA()
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# # tensorrt
include_directories(/home/jjho/software/TensorRT-7.1.3.4/include/)
link_directories(/home/jjho/software/TensorRT-7.1.3.4/lib/)

if (USE_CUDA)
  message("CUDA is available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND OR INCOMPATIBLE CMAKE VERSION FOUND")
  set(CUDA_AVAIL OFF)
endif ()

# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER libnvinfer.so /home/jjho/software/TensorRT-7.1.3.4/lib)
find_library(NVPARSERS libnvparsers.so /home/jjho/software/TensorRT-7.1.3.4/lib)
find_library(NVONNXPARSERS libnvonnxparser.so /home/jjho/software/TensorRT-7.1.3.4/lib)
# try to find the tensorRT modules
# find_library(NVINFER libnvinfer.so)
# find_library(NVPARSERS libnvparsers.so)
# find_library(NVONNXPARSERS libnvonnxparser.so)

if(NVINFER AND NVPARSERS AND NVONNXPARSERS)
  message("TensorRT is available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSERS: ${NVONNXPARSERS}")
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
  libcudnn.so libcudnn_static.a
  # NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  # PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  # PATH_SUFFIXES lib lib64 bin
  # DOC "CUDNN library."
  /usr/local/cuda-10.2/lib64
)

if(CUDNN_LIBRARY)
  message("CUDNN is available!")
  message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()

if(TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    /usr/local/cuda-10.2/include
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
  )

  set(SOURCE_FILES
    src/main.cpp
    src/livox_detection.cpp
    src/clip.cpp
  )

  add_executable(livox_detection
    ${SOURCE_FILES}
  )

  cuda_add_library(gpu_livox_detection_lib
    src/postprocess_cuda.cu
    src/iou3d_nms_kernel.cu
  )

  target_link_libraries(gpu_livox_detection_lib
    ${CUDA_LIBRARIES}
  )

  target_link_libraries(livox_detection
    ${NVINFER}
    ${NVONNXPARSERS}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
    ${Eigen3_LIBRARIES}
    ${PCL_LIBRARIES}
    gpu_livox_detection_lib
  )


endif()
