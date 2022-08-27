// 自己定义的头文件
#include "livox_detection.hpp"

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>
#include <cstring> //必须引用
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <boost/shared_ptr.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

void inference()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("livox.pcd", *input_cloud) == -1)
    {
        std::cerr << "open failed!" << std::endl;
    }

    livox_detection livox;

    livox.doprocess(input_cloud);
}

int main()
{
    if (!build_model())
    {
        return -1;
    }
    inference();
    return 0;
}