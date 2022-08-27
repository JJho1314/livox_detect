
// headers in CUDA
#include <thrust/sort.h>

// headers in local files
#include "postprocess_cuda.h"

__global__ void filter_kernel(const float *all_preds, float *filtered_box, float *filtered_score, int *filter_label, int *filter_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int counter = atomicAdd

        float box_px = all_preds[tid * 9 + 0];
    float box_py = all_preds[tid * 9 + 1];
    float box_pz = all_preds[tid * 9 + 2];
    float box_dx = all_preds[tid * 9 + 3];
    float box_dy = all_preds[tid * 9 + 4];
    float box_dz = all_preds[tid * 9 + 5];
    float box_theta = all_preds[tid * 9 + 6];
    float box_score = all_preds[tid * 9 + 7];
    float box_cls = all_preds[tid * 9 + 8];

    if (box_px > cloud_x_min && box_px < cloud_x_max && box_py > cloud_y_min && box_py < cloud_y_max && box_pz > cloud_z_min && box_pz < cloud_z_max && box_score > score_thresh[box.cls])
    {
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 0] = box_px;
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 1] = box_py;
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 2] = box_pz;
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 3] = box_dx;
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 4] = box_dy;
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 5] = box_dz;
        filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 6] = box_theta;

        filtered_score[counter] = box_score;
        filtered_label[counter] = box_cls;
    }
}

void PostprocessCuda::doPostprocessCuda(const float *rpn_all_output, float *filtered_box, float *filtered_score, int *filter_label, int *filter_count)
{
    const int num_blocks_filter_kernel = DIVUP(num_anchor_, NUM_THREADS_); // Number of threads when launching cuda kernel
    filter_kernel<<<num_blocks_filter_kernel, NUM_THREADS_>>>(rpn_all_output, filtered_box, filtered_score, filter_label, filter_count);

    int boxSize = findValidScoreNum(filtered_score, SCORE_THREAHOLD, 448, 1120); //用于设置阈值控制分数

    int boxSizeAft = nms_gpu(filtered_box, filtered_score, boxSize, NMS_THREAHOLD);

    GPU_CHECK(cudaMemcpy(host_score_indexs, dev_score_indexs, boxSize * sizeof(int), cudaMemcpyDeviceToHost));
    for (auto i = 0; i < boxSizeAft; i++)
    {
        int ii = host_keep_data[i];
        // std::cout <<i<< ", "<<ii<<", \n";
        int idx = host_score_indexs[ii];
        int xIdx = idx % OUTPUT_W;
        int yIdx = idx / OUTPUT_W;
        Box box;

        box.x = (host_boxes[i + 0 * boxSizeAft] + xIdx) * OUT_SIZE_FACTOR * X_STEP + X_MIN;
        box.y = (host_boxes[i + 1 * boxSizeAft] + yIdx) * OUT_SIZE_FACTOR * Y_STEP + Y_MIN;
        box.z = host_boxes[i + 2 * boxSizeAft];
        box.l = host_boxes[i + 3 * boxSizeAft];
        box.h = host_boxes[i + 4 * boxSizeAft];
        box.w = host_boxes[i + 5 * boxSizeAft];
        float theta_s = host_boxes[i + 6 * boxSizeAft];
        float theta_c = host_boxes[i + 7 * boxSizeAft];
        box.theta = atan2(theta_s, theta_c);
        box.score = host_boxes[i + 8 * boxSizeAft];
        box.cls = host_label[i];
        box.velX = idx;
        box.velY = 0;
        predResult.push_back(box);
    }
}
