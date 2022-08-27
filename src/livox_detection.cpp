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
#include <string>

#include <eigen3/Eigen/Core>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template <typename _T>
std::shared_ptr<_T> make_nvshared(_T *ptr)
{
    return std::shared_ptr<_T>(ptr, [](_T *p)
                               { p->destroy(); });
}

bool exists(const std::string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

std::vector<unsigned char> load_file(const std::string &file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

// 上一节的代码
bool build_model()
{
    if (exists("Livox_detection_sim.engine"))
    {
        printf("Engine file has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile("Livox_detection_sim.onnx", 1))
    {
        printf("Failed to parse onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    int maxBatchSize = 1;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    // 配置最小、最优、最大范围
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE *f = fopen("Livox_detection_sim.engine", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

livox_detection::livox_detection()
{
    // checkRuntime(cudaMalloc(&head_buffers_[0], PRE_BOX_SIZE * sizeof(float)));
    // checkRuntime(cudaMalloc(&head_buffers_[1], PRE_SCORES_SIZE * sizeof(float)));
    // checkRuntime(cudaMalloc(&head_buffers_[2], PRED_LABELS_SIZE * sizeof(float)));
}

inline void livox_detection::point_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const int min, const int max, std::string axis, bool setFilterLimitsNegative)
{
    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud);
    filter.setFilterFieldName(axis);

    filter.setFilterLimits(min, max);

    if (setFilterLimitsNegative == true)
    {
        filter.setFilterLimitsNegative(true);
    }

    filter.filter(*cloud);
}

void livox_detection::mask_points_out_of_range(pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr)
{
    pcl::PointXYZ min;
    pcl::PointXYZ max;

    point_filter(in_pcl_pc_ptr, cloud_x_min, cloud_x_max - 0.01, "x", false);
    point_filter(in_pcl_pc_ptr, cloud_y_min, cloud_y_max - 0.01, "y", false);
    point_filter(in_pcl_pc_ptr, cloud_z_min, cloud_z_max - 0.01, "z", false);

    pcl::getMinMax3D(*in_pcl_pc_ptr, min, max);

    std::cout << max.x << "," << min.x << std::endl;
    std::cout << max.y << "," << min.y << std::endl;
    std::cout << max.z << "," << min.z << std::endl;
}

void livox_detection::pclToArray(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, float *out_points_array)
{
    float DX = voxel_size[0];
    float DY = voxel_size[1];
    float DZ = voxel_size[2];

    for (size_t i = 0; i < in_pcl_pc_ptr->size(); i++)
    {
        pcl::PointXYZ point = in_pcl_pc_ptr->at(i);
        int pc_lidar_x = floor((point.x - cloud_x_min) / DX);
        int pc_lidar_y = floor((point.y - cloud_y_min) / DY);
        int pc_lidar_z = floor((point.z - cloud_z_min) / DZ);
        out_points_array[BEV_W * BEV_H * pc_lidar_z + BEV_W * pc_lidar_y + pc_lidar_x] = 1;
    }
}

void livox_detection::preprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_pcl_pc_ptr, float *out_points_array)
{
    std::cout << "livox detect preprocess start" << std::endl;

    float theta = 0;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX())); //同理，UnitX(),绕X轴；UnitY(),绕Y轴
    transform.translation() << 0.0, 0.0, offset_ground;                   // 三个数分别对应X轴、Y轴、Z轴方向上的平移

    pcl::transformPointCloud(*in_pcl_pc_ptr, *out_pcl_pc_ptr, transform);

    mask_points_out_of_range(out_pcl_pc_ptr);

    pclToArray(out_pcl_pc_ptr, out_points_array);

    std::cout << "livox detect preprocess finish" << std::endl;
}

void livox_detection::doprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr)
{
    TRTLogger logger;
    auto engine_data = load_file("Livox_detection_sim.engine");
    auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_numel = input_batch * BEV_C * BEV_H * BEV_W;
    float *input_data_host;
    float *input_data_device;

    float *points_array = new float[input_numel];

    for (int i = 0; i < input_numel; i++)
    {
        points_array[i] = 0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    preprocess(in_pcl_pc_ptr, transformed_cloud_ptr, points_array);

    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    input_data_host = points_array;

    clock_t start = clock();

    ///////////////////////////////////////////////////
    // image to float

    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    float *output_data_host;
    float *output_data_device;
    checkRuntime(cudaMallocHost(&output_data_host, OUTPUT_SIZE * sizeof(float)));
    checkRuntime(cudaMalloc(&output_data_device, OUTPUT_SIZE * sizeof(float)));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    // 设置当前推理时，input大小
    execution_context->setBindingDimensions(0, input_dims);
    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    checkRuntime(cudaStreamDestroy(stream));

    clock_t end = clock();

    // std::vector<Box> Box_Vehicle;
    // std::vector<Box> Box_Pedestrian_before;
    // std::vector<Box> Box_Cyclist_before;

    // for (int i = 0; i < 500; i++)
    // {
    //     float *ptr = output_data_host + i * 9;
    //     Box box;
    //     box.x = ptr[0];
    //     box.y = ptr[1];
    //     box.z = ptr[2];
    //     box.dx = ptr[3];
    //     box.dy = ptr[4];
    //     box.dz = ptr[5];
    //     box.theta = ptr[6];
    //     box.score = ptr[7];
    //     box.cls = ptr[8];

    //     if (box.x > cloud_x_min && box.x < cloud_x_max && box.y > cloud_y_min && box.y < cloud_y_max && box.z > cloud_z_min && box.z < cloud_z_max && box.score > score_thresh[box.cls])
    //     {
    //         if (box.cls == 0)
    //         {
    //             Box_Vehicle.push_back(box);
    //         }
    //         else if (box.cls == 1)
    //         {
    //             Box_Pedestrian_before.push_back(box);
    //         }
    //         else if (box.cls == 2)
    //         {
    //             Box_Cyclist_before.push_back(box);
    //         }
    //     }
    // }

    printf("Total time: %lf s \n", (double)(end - start) / CLOCKS_PER_SEC);

    std::cout << "livox detect infer finish" << std::endl;

    delete[] points_array;
    // checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));

    // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("cloud"));
    // viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud_ptr, "sample cloud");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    // viewer->addCoordinateSystem(1.0);
    // viewer->initCameraParameters();

    // viewer->setBackgroundColor(0, 0, 0);

    // for (int i = 0; i < Box_Vehicle.size(); i++)
    // {
    //     std::string name = "Vehicle" + std::to_string(i);
    //     viewer->addCube(float(Box_Vehicle[i].x) - Box_Vehicle[i].dx / 2, Box_Vehicle[i].x + Box_Vehicle[i].dx / 2, float(Box_Vehicle[i].y) - Box_Vehicle[i].dy / 2, Box_Vehicle[i].y + Box_Vehicle[i].dy / 2, float(Box_Vehicle[i].z) - Box_Vehicle[i].dz / 2, Box_Vehicle[i].z + Box_Vehicle[i].dz / 2, 1.0, 1.0, 1.0, name);
    //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, name); //绿框
    //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, name);

    //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, name);
    //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, name);
    // }

    // while (!viewer->wasStopped())
    // {
    //     viewer->spinOnce(100);
    //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }
}

void livox_detection::postprocess(const float *in_points_array)
{
}

livox_detection::~livox_detection()
{
}
