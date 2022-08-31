#ifndef LIVOX_DETECTION
#define LIVOX_DETECTION

#include "postprocess_cuda.h"

#include <iostream>
// 推理用的运行时头文件
#include <NvInferRuntime.h>

#include <pcl/common/common.h>

// headers in ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

bool build_model();

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 httpsoutput_data_host://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }

    inline const char *severity_string(nvinfer1::ILogger::Severity t)
    {
        switch (t)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:
            return "error";
        case nvinfer1::ILogger::Severity::kWARNING:
            return "warning";
        case nvinfer1::ILogger::Severity::kINFO:
            return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE:
            return "verbose";
        default:
            return "unknow";
        }
    }
};

class livox_detection
{
public:
    livox_detection();

    ~livox_detection();

    /**
     * @brief Create ROS pub/sub obejct
     * @details Create/Initializing ros pub/sub object
     */
    void createROSPubSub();

    int BEV_W = 1120;
    int BEV_H = 448;
    int BEV_C = 30;

private:
    void point_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const int min, const int max, std::string axis, bool setFilterLimitsNegative);

    void pclToArray(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, float *out_points_array);

    void mask_points_out_of_range(pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr);

    void preprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_pcl_pc_ptr, float *out_points_array);

    void doprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr);

    void postprocess(const float *rpn_all_output, std::vector<Box> &predResult);

    void initTRT();

    /**
     * @brief callback for pointcloud
     * @param[in] input pointcloud from lidar sensor
     * @details Call point_pillars to get 3D bounding box
     */
    void pointsCallback(const sensor_msgs::PointCloud2::ConstPtr &input);

    // initializer list
    ros::NodeHandle private_nh_;

    ros::NodeHandle nh_;
    ros::Subscriber sub_points_;
    ros::Publisher pub_objects_;
    ros::Publisher pub_objects_marker_;

    std::vector<unsigned char> engine_data;
    nvinfer1::IExecutionContext *execution_context;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;

    float *input_data_host;
    float *input_data_device;
    float *output_data_host;
    float *output_data_device;

    float *dev_filtered_box_;
    float *dev_filtered_score_;
    int *dev_filtered_label_;
    int dev_filter_count_;
    long *dev_keep_data_;

    float theta = 0;
    int input_batch = 1;
    int input_numel = input_batch * BEV_C * BEV_H * BEV_W;
    const int OUTPUT_SIZE = 1 * 9 * 500;
    float point_cloud_range[6] = {0, -44.8, -2, 224, 44.8, 4};
    float voxel_size[3] = {0.2, 0.2, 0.2};
};

#endif