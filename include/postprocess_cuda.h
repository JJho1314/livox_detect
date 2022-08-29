#ifndef POSTPROCESS_CUDA_H
#define POSTPROCESS_CUDA_H

#include "config.h"
// headers in STL
#include <memory>
#include <iostream>

#include <vector>

#include "common.h"

struct Box
{
    float x;
    float y;
    float z;
    float dx;
    float dy;
    float dz;
    float theta;
    float score;
    int cls;
    bool isDrop; // for nms
};

int findValidScoreNum(float *score, float thre, int num_anchor);

class PostprocessCuda
{
private:
    const int num_anchor_;
    const int num_class_;
    const int NUM_OUTPUT_BOX_FEATURE_;
    float score_thresh_[3];

public:
    PostprocessCuda(const int num_anchor,
                    const int num_class,
                    const int num_output_box_feature,
                    float *score_thresh);

    void doPostprocessCuda(const float *rpn_all_output, float *dev_filtered_box, float *dev_filtered_score, int *dev_filter_label, int &dev_filter_count, long *dev_keep_data, std::vector<Box> &predResult);
};

#endif // POSTPROCESS_CUDA_H