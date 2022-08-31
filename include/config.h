#ifndef __CENTERNET_CONFIG_H__
#define __CENTERNET_CONFIG_H__

// voxel size
#define offset_ground 1.8
#define cloud_x_min 0
#define cloud_x_max 224
#define cloud_y_min -44.8
#define cloud_y_max 44.8
#define cloud_z_min -2
#define cloud_z_max 4

#define PI 3.141592653f

// paramerters for postprocess
#define SCORE_THREAHOLD 0.2f
#define NMS_THREAHOLD 0.3f
#define INPUT_NMS_MAX_SIZE 4096
#define OUTPUT_NMS_MAX_SIZE 500
// #define THREADS_PER_BLOCK_NMS  sizeof(unsigned long long) * 8
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

// OUT_SIZE_FACTOR * OUTPUT_H  * Y_STEP = Y_MAX - Y_MIN
#define OUT_SIZE_FACTOR 1.0f

#define DIM_CHANNEL 3

#define NUM_CLASS_ 3
#define NUM_ANCHOR 500
#define ANCHOR_SIZE 11
#define NUM_OUTPUT_BOX_FEATURE 7
// const int OUTPUT_SIZE = 1 * 11 * 448 * 1120;
#define NUM_ANCHOR_ 500

#endif
