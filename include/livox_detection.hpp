#ifndef LIVOX_DETECTION
#define LIVOX_DETECTION

#include <iostream>
// 推理用的运行时头文件
#include <NvInferRuntime.h>

#include <pcl/common/common.h>

#include "postprocess_cuda.h"

bool build_model();

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
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

    void preprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_pcl_pc_ptr, float *out_points_array);

    void doprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr);

    void postprocess(const float *rpn_all_output, std::vector<Box> &predResult);

    int BEV_W = 1120;
    int BEV_H = 448;
    int BEV_C = 30;

private:
    void point_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const int min, const int max, std::string axis, bool setFilterLimitsNegative);

    void pclToArray(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, float *out_points_array);

    void mask_points_out_of_range(pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr);

    pcl::PointCloud<pcl::PointXYZ> input_cloud_;

    const int OUTPUT_SIZE = 1 * 9 * 500;
    float *dev_filtered_box_;
    float *dev_filtered_score_;
    int *dev_filtered_label_;
    int dev_filter_count_;
    long *dev_keep_data_;
    float point_cloud_range[6] = {0, -44.8, -2, 224, 44.8, 4};
    float score_thresh[3] = {0.2, 0.3, 0.3};
    float voxel_size[3] = {0.2, 0.2, 0.2};
};

#endif