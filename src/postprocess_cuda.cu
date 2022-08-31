
// headers in CUDA
#include <thrust/sort.h>
#include <common.h>

// headers in local files
#include "livox_detection.hpp"
#include "postprocess_cuda.h"

#include "iou3d_nms.h"

__global__ void filter_kernel(const float *rpn_all_output, float *filtered_box, float *filtered_score, int *filtered_label, int *filter_count, const int num_output_box_feature, float *score_thresh, int num_anchor)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("GPU thread info X:%d Y:%d Z:%d\t block info X:%d Y:%d Z:%d\n",
    //        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    printf("GPU tid : %d\n", tid);
    if (tid < num_anchor)
    {
        float box_px = rpn_all_output[tid * 9 + 0];
        float box_py = rpn_all_output[tid * 9 + 1];
        float box_pz = rpn_all_output[tid * 9 + 2];
        float box_dx = rpn_all_output[tid * 9 + 3];
        float box_dy = rpn_all_output[tid * 9 + 4];
        float box_dz = rpn_all_output[tid * 9 + 5];
        float box_theta = rpn_all_output[tid * 9 + 6];
        float box_score = rpn_all_output[tid * 9 + 7];
        int box_cls = rpn_all_output[tid * 9 + 8];

        if (box_px > 0 && box_px < 224 && box_py > -44.8 && box_py < 44.8 && box_pz > -2 && box_pz < 4 && box_score > score_thresh[box_cls])
        {
            filtered_box[tid * num_output_box_feature + 0] = box_px;
            filtered_box[tid * num_output_box_feature + 1] = box_py;
            filtered_box[tid * num_output_box_feature + 2] = box_pz;
            filtered_box[tid * num_output_box_feature + 3] = box_dx;
            filtered_box[tid * num_output_box_feature + 4] = box_dy;
            filtered_box[tid * num_output_box_feature + 5] = box_dz;
            filtered_box[tid * num_output_box_feature + 6] = box_theta;

            filtered_score[tid] = box_score;
            filtered_label[tid] = box_cls;

            // printf("box_cls:(%f)\n", box_cls);
        }
    }
}

PostprocessCuda::PostprocessCuda(const int num_anchor,
                                 const int num_class,
                                 const int num_output_box_feature,
                                 float *score_thresh) : num_anchor_(num_anchor), num_class_(num_class), NUM_OUTPUT_BOX_FEATURE_(num_output_box_feature)
{
    // GPU_CHECK(cudaMalloc(&score_thresh_, 3 * sizeof(float)));
    score_thresh_[0] = score_thresh[0];
    score_thresh_[1] = score_thresh[1];
    score_thresh_[2] = score_thresh[2];
}

void PostprocessCuda::doPostprocessCuda(const float *rpn_all_output, float *dev_filtered_box, float *dev_filtered_score, int *dev_filter_label, int &dev_filter_count, long *dev_keep_data, std::vector<Box> &predResult)
{
    // int NUM_THREADS_ = 64;
    // const int num_blocks_filter_kernel = DIVUP(num_anchor_, NUM_THREADS_); // Number of threads when launching cuda kernel
    // filter_kernel<<<num_blocks_filter_kernel, NUM_THREADS_>>>(rpn_all_output, dev_filtered_box, dev_filtered_score, dev_filter_label, dev_filter_count, NUM_OUTPUT_BOX_FEATURE_, score_thresh_, num_anchor_);

    // printf("box_cls:(%f)\n", box_cls);
    float score_thresh[3] = {0.2, 0.3, 0.3};
    int boxSize = findValidScoreNum(dev_filtered_score, SCORE_THREAHOLD, num_anchor_); //用于设置阈值控制分数

    int boxSizeAft = nms_gpu(dev_filtered_box, dev_keep_data, boxSize, NMS_THREAHOLD);

    // float *host_keep_data = nullptr;

    // GPU_CHECK(cudaMemcpy(host_keep_data, dev_keep_data, boxSize * sizeof(int), cudaMemcpyDeviceToHost));
    for (auto i = 0; i < boxSizeAft; i++)
    {
        int ii = dev_keep_data[i];
        // std::cout << i << ", " << ii << ", \n";
        int idx = ii;
        Box box;

        box.x = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 0];
        box.y = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 1];
        box.z = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 2];
        box.dx = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 3];
        box.dy = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 4];
        box.dz = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 5];
        box.theta = dev_filtered_box[idx * NUM_OUTPUT_BOX_FEATURE_ + 6];
        box.score = dev_filtered_score[idx];
        box.cls = dev_filter_label[idx];
        if (box.x > cloud_x_min && box.x < cloud_x_max && box.y > cloud_y_min && box.y < cloud_y_max && box.z > cloud_z_min && box.z < cloud_z_max && box.score > score_thresh[box.cls])
        {
            predResult.push_back(box);
        }
    }
}
