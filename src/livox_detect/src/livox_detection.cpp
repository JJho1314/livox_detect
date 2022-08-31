// 自己定义的头文件
#include "livox_detection.hpp"
#include "clip.h"

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

// headers in local files
#include "autoware_msgs/DetectedObjectArray.h"

// headers in ROS
#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>

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

void livox_detection::initTRT()
{
    TRTLogger logger;
    engine_data = load_file(engine_Path.c_str());
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    execution_context = (engine->createExecutionContext());
}

livox_detection::livox_detection() : private_nh_("~")
{
    private_nh_.param<std::string>("engine_Path", engine_Path, "");
    initTRT();
}

void livox_detection::point_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const int min, const int max, std::string axis, bool setFilterLimitsNegative)
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
    // pcl::PointXYZ min;
    // pcl::PointXYZ max;

    point_filter(in_pcl_pc_ptr, cloud_x_min, cloud_x_max - 0.01, "x", false);
    point_filter(in_pcl_pc_ptr, cloud_y_min, cloud_y_max - 0.01, "y", false);
    point_filter(in_pcl_pc_ptr, cloud_z_min - offset_ground, cloud_z_max - offset_ground - 0.01, "z", false);

    // pcl::getMinMax3D(*in_pcl_pc_ptr, min, max);

    // std::cout << max.x << "," << min.x << std::endl;
    // std::cout << max.y << "," << min.y << std::endl;
    // std::cout << max.z << "," << min.z << std::endl;
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
        int pc_lidar_z = floor((point.z + offset_ground - cloud_z_min) / DZ);
        out_points_array[BEV_W * BEV_H * pc_lidar_z + BEV_W * pc_lidar_y + pc_lidar_x] = 1;
    }
}

void livox_detection::preprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_pcl_pc_ptr, float *out_points_array)
{
    std::cout << "livox detect preprocess start" << std::endl;

    Clip clip(nh_, private_nh_);

    clip.Process(*in_pcl_pc_ptr, *out_pcl_pc_ptr); //删除多余的点

    // mask_points_out_of_range(out_pcl_pc_ptr);

    pclToArray(out_pcl_pc_ptr, out_points_array);

    std::cout << "livox detect preprocess finish" << std::endl;
}

void livox_detection::postprocess(const float *rpn_all_output, std::vector<Box> &predResult)
{
    PostprocessCuda postprococess(NUM_ANCHOR, NUM_CLASS_, NUM_OUTPUT_BOX_FEATURE);

    checkRuntime(cudaMallocHost(&dev_filtered_box_, NUM_ANCHOR_ * NUM_OUTPUT_BOX_FEATURE * sizeof(float)));
    checkRuntime(cudaMallocHost(&dev_filtered_score_, NUM_ANCHOR_ * sizeof(float)));
    checkRuntime(cudaMallocHost(&dev_filtered_label_, NUM_ANCHOR_ * sizeof(int)));
    checkRuntime(cudaMallocHost(&dev_keep_data_, NUM_ANCHOR_ * sizeof(long)));

    dev_filter_count_ = 0;
    for (int i = 0; i < NUM_ANCHOR_; i++)
    {
        float box_px = rpn_all_output[i * 9 + 0];
        float box_py = rpn_all_output[i * 9 + 1];
        float box_pz = rpn_all_output[i * 9 + 2];
        float box_dx = rpn_all_output[i * 9 + 3];
        float box_dy = rpn_all_output[i * 9 + 4];
        float box_dz = rpn_all_output[i * 9 + 5];
        float box_theta = rpn_all_output[i * 9 + 6];
        float box_score = rpn_all_output[i * 9 + 7];
        int box_cls = rpn_all_output[i * 9 + 8];

        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 0] = box_px;
        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 1] = box_py;
        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 2] = box_pz;
        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 3] = box_dx;
        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 4] = box_dy;
        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 5] = box_dz;
        dev_filtered_box_[dev_filter_count_ * NUM_OUTPUT_BOX_FEATURE + 6] = box_theta;

        dev_filtered_score_[dev_filter_count_] = box_score;
        dev_filtered_label_[dev_filter_count_] = box_cls;

        dev_filter_count_++;
    }

    postprococess.doPostprocessCuda(rpn_all_output, dev_filtered_box_, dev_filtered_score_, dev_filtered_label_, dev_filter_count_, dev_keep_data_, predResult);
    checkRuntime(cudaFreeHost(dev_filtered_box_));
    checkRuntime(cudaFreeHost(dev_filtered_score_));
    checkRuntime(cudaFreeHost(dev_filtered_label_));
    checkRuntime(cudaFreeHost(dev_keep_data_));
}

void livox_detection::doprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_pcl_pc_ptr, std::vector<Box> &pre_box)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    preprocess(in_pcl_pc_ptr, transformed_cloud_ptr, input_data_host);

    ///////////////////////////////////////////////////
    // image to float

    ///////////////////////////////////////////////////

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

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

    postprocess(output_data_host, pre_box);

    checkRuntime(cudaStreamDestroy(stream));

    std::cout << "livox detect infer finish" << std::endl;
}

geometry_msgs::Pose livox_detection::getTransformedPose(const geometry_msgs::Pose &in_pose, const tf::Transform &tf)
{
    tf::Transform transform;
    geometry_msgs::PoseStamped out_pose;
    transform.setOrigin(tf::Vector3(in_pose.position.x, in_pose.position.y, in_pose.position.z));
    transform.setRotation(
        tf::Quaternion(in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w));
    geometry_msgs::PoseStamped pose_out;
    tf::poseTFToMsg(tf * transform, out_pose.pose);
    return out_pose.pose;
}

void livox_detection::pubDetectedObject_Marker(const std::vector<Box> &detections, const std_msgs::Header &in_header)
{
    autoware_msgs::DetectedObjectArray objects;
    objects.header = in_header;

    // clear all markers before
    visualization_msgs::MarkerArray empty_markers;
    visualization_msgs::Marker clear_marker;
    clear_marker.header = in_header;
    clear_marker.ns = "objects";
    clear_marker.id = 0;
    clear_marker.action = clear_marker.DELETEALL;
    clear_marker.lifetime = ros::Duration();
    empty_markers.markers.push_back(clear_marker);
    pub_objects_marker_.publish(empty_markers);

    visualization_msgs::MarkerArray object_markers;

    for (size_t i = 0; i < detections.size(); i++)
    {
        /*Autoware start*/
        autoware_msgs::DetectedObject object;
        object.header = in_header;
        object.valid = true;
        object.pose_reliable = true;

        object.pose.position.x = detections[i].x;
        object.pose.position.y = detections[i].y;
        object.pose.position.z = detections[i].z - offset_ground;

        // Trained this way
        float autoware_yaw = detections[i].theta;
        autoware_yaw += M_PI / 2;
        autoware_yaw = std::atan2(std::sin(autoware_yaw), std::cos(autoware_yaw));
        geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(autoware_yaw);
        object.pose.orientation = q;

        // if (true)
        // {
        //     object.pose = getTransformedPose(object.pose, angle_transform_inversed_);
        // }

        // Again: Trained this way
        object.dimensions.x = detections[i].dx;
        object.dimensions.y = detections[i].dy;
        object.dimensions.z = detections[i].dz;

        /*Auwoware Msg End*/

        const float object_px = detections[i].x;
        const float object_py = detections[i].y;
        const float object_pz = detections[i].z - offset_ground;
        const float object_dx = detections[i].dx;
        const float object_dy = detections[i].dy;
        const float object_dz = detections[i].dz;
        const float object_yaw = detections[i].theta;

        float yaw = std::atan2(std::sin(object_yaw), std::cos(object_yaw));
        float cos_yaw = std::cos(yaw);
        float sin_yaw = std::sin(yaw);
        float half_dx = object_dx / 2;
        float half_dy = object_dy / 2;
        float half_dz = object_dz / 2;

        // for autoware
        //  geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(yaw+M_PI/2);
        //  object.pose.orientation = q;

        visualization_msgs::Marker box, dir_arrow, text_show;
        box.header = dir_arrow.header = text_show.header = in_header;
        box.ns = dir_arrow.ns = text_show.ns = "objects";
        box.id = i;
        // dir_arrow.id = obj + objects_array.size();
        text_show.id = i + detections.size();
        box.type = visualization_msgs::Marker::LINE_LIST;
        dir_arrow.type = visualization_msgs::Marker::ARROW;
        text_show.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        geometry_msgs::Point p[24];
        // Ground side
        // A->B
        p[0].x = object_px + half_dx * cos_yaw - half_dy * sin_yaw;
        p[0].y = object_py + half_dx * sin_yaw + half_dy * cos_yaw;
        p[0].z = object_pz - half_dz;
        p[1].x = object_px + half_dx * cos_yaw + half_dy * sin_yaw;
        p[1].y = object_py + half_dx * sin_yaw - half_dy * cos_yaw;
        p[1].z = object_pz - half_dz;
        // B->C
        p[2].x = p[1].x;
        p[2].y = p[1].y;
        p[2].z = p[1].z;
        p[3].x = object_px - half_dx * cos_yaw + half_dy * sin_yaw;
        p[3].y = object_py - half_dx * sin_yaw - half_dy * cos_yaw;
        p[3].z = object_pz - half_dz;
        // C->D
        p[4].x = p[3].x;
        p[4].y = p[3].y;
        p[4].z = p[3].z;
        p[5].x = object_px - half_dx * cos_yaw - half_dy * sin_yaw;
        p[5].y = object_py - half_dx * sin_yaw + half_dy * cos_yaw;
        p[5].z = object_pz - half_dz;
        // D->A
        p[6].x = p[5].x;
        p[6].y = p[5].y;
        p[6].z = p[5].z;
        p[7].x = p[0].x;
        p[7].y = p[0].y;
        p[7].z = p[0].z;

        // Top side
        // E->F
        p[8].x = p[0].x;
        p[8].y = p[0].y;
        p[8].z = object_pz + half_dz;
        p[9].x = p[1].x;
        p[9].y = p[1].y;
        p[9].z = object_pz + half_dz;
        // F->G
        p[10].x = p[1].x;
        p[10].y = p[1].y;
        p[10].z = object_pz + half_dz;
        p[11].x = p[3].x;
        p[11].y = p[3].y;
        p[11].z = object_pz + half_dz;
        // G->H
        p[12].x = p[3].x;
        p[12].y = p[3].y;
        p[12].z = object_pz + half_dz;
        p[13].x = p[5].x;
        p[13].y = p[5].y;
        p[13].z = object_pz + half_dz;
        // H->E
        p[14].x = p[5].x;
        p[14].y = p[5].y;
        p[14].z = object_pz + half_dz;
        p[15].x = p[0].x;
        p[15].y = p[0].y;
        p[15].z = object_pz + half_dz;

        // Around side
        // A->E
        p[16].x = p[0].x;
        p[16].y = p[0].y;
        p[16].z = p[0].z;
        p[17].x = p[8].x;
        p[17].y = p[8].y;
        p[17].z = p[8].z;
        // B->F
        p[18].x = p[1].x;
        p[18].y = p[1].y;
        p[18].z = p[1].z;
        p[19].x = p[9].x;
        p[19].y = p[9].y;
        p[19].z = p[9].z;
        // C->G
        p[20].x = p[3].x;
        p[20].y = p[3].y;
        p[20].z = p[3].z;
        p[21].x = p[11].x;
        p[21].y = p[11].y;
        p[21].z = p[11].z;
        // D->H
        p[22].x = p[5].x;
        p[22].y = p[5].y;
        p[22].z = p[5].z;
        p[23].x = p[13].x;
        p[23].y = p[13].y;
        p[23].z = p[13].z;

        for (size_t pi = 0u; pi < 24; ++pi)
        {
            box.points.push_back(p[pi]);
        }
        box.scale.x = 0.1;
        // box.color = color;
        box.color.a = 1.0;
        box.color.r = 0.18;
        box.color.g = 0.45;
        box.color.b = 0.70;

        object_markers.markers.push_back(box);

        // text
        geometry_msgs::Pose pose;
        pose.position.x = object_px;
        pose.position.y = object_py;
        pose.position.z = object_pz + half_dz;
        text_show.pose = pose;

        std::ostringstream str;
        str.precision(2);
        str.setf(std::ios::fixed);
        // double vx = objects_array[obj]->velocity[0];
        // double vy = objects_array[obj]->velocity[1];
        // double ground_v = sqrt(vx * vx + vy * vy);
        // double distance_ck = sqrt(center(0)*center(0) + center(1)*center(1));
        // str << "id:"<< objects_array[obj]->track_id<<"\n" ;//<< "v:" << ground_v;
        // str << "d: "<<distance_ck<<"\n";
        // str << "v:" << ground_v;

        // add label
        if (detections[i].cls == 0)
        {
            str << " car";
        }
        else if (detections[i].cls == 1)
        {
            str << " pedestrian";
        }
        else if (detections[i].cls == 2)
        {
            str << " cyclist";
        }
        else
        {
            str << "car";
            // printf("Why output unknown object ?!\n");
        }

        text_show.text = str.str();

        text_show.action = visualization_msgs::Marker::ADD;
        text_show.color.a = 1.0;
        text_show.color.r = 1.0;
        text_show.color.g = 1.0;
        text_show.color.b = 0.0;

        text_show.scale.z = 1;
        object_markers.markers.push_back(text_show);

        // for autoware object label
        object.label = str.str();
        objects.objects.push_back(object);
    }
    pub_objects_marker_.publish(object_markers);
    pub_objects_.publish(objects);
}

void livox_detection::pointsCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcl_pc_ptr);
    double start_time = ros::Time::now().toSec();

    // 去除128中的nan点
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < pcl_pc_ptr->size(); i++)
    {
        pcl::PointXYZ point = pcl_pc_ptr->at(i);
        if (std::isnan(point.x) || std::isinf(point.x) || std::isnan(point.y) || std::isinf(point.y) ||
            std::isnan(point.z) || std::isinf(point.z))
        {
            continue;
        }
        filtered_cloud_ptr->push_back(point);
    }
    pcl_pc_ptr = filtered_cloud_ptr;

    // 去除128中的nan点
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX())); //同理，UnitX(),绕X轴；UnitY(),绕Y轴
    pcl::transformPointCloud(*pcl_pc_ptr, *out_pcl_pc_ptr, transform);

    std::vector<Box> pre_box;
    doprocess(out_pcl_pc_ptr, pre_box);

    pubDetectedObject_Marker(pre_box, msg->header);
}

void livox_detection::createROSPubSub()
{
    sub_points_ = nh_.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, &livox_detection::pointsCallback, this);
    pub_objects_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector_3d/objects", 1);
    pub_objects_marker_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection/pointpillars_objects", 1);
}

livox_detection::~livox_detection()
{
    execution_context->destroy();
    runtime->destroy();
    engine->destroy();

    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}
