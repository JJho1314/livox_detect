#ifndef POSTPROCESS_CUDA_H
#define POSTPROCESS_CUDA_H

// headers in STL
#include <memory>
#include <iostream>

#include <vector>

#include <nms_cuda.h>

int findValidScoreNum(float *score, float thre, int output_h, int output_w);

class PostprocessCuda
{
private:
    const float FLOAT_MIN_;
    const float FLOAT_MAX_;
    const int NUM_ANCHOR_X_INDS_;
    const int NUM_ANCHOR_Y_INDS_;
    const int NUM_ANCHOR_R_INDS_;
    const int num_anchor_;
    const int num_class_;
    const int num_head_;
    const int Rpn_Cls_Output_Dim_;
    const float score_threshold_;
    const int NUM_THREADS_;
    const float nms_overlap_threshold_;
    const int NUM_BOX_CORNERS_;
    const int NUM_OUTPUT_BOX_FEATURE_;

    std::unique_ptr<NMSCuda> nms_cuda_ptr_;

public:
    /**
     * @brief Constructor
     * @param[in] FLOAT_MIN The lowest float value
     * @param[in] FLOAT_MAX The maximum float value
     * @param[in] NUM_ANCHOR_X_INDS Number of x-indexes for anchors
     * @param[in] NUM_ANCHOR_Y_INDS Number of y-indexes for anchors
     * @param[in] NUM_ANCHOR_R_INDS Number of rotation-indexes for anchors
     * @param[in] score_threshold Score threshold for filtering output
     * @param[in] NUM_THREADS Number of threads when launching cuda kernel
     * @param[in] nms_overlap_threshold IOU threshold for NMS
     * @param[in] NUM_BOX_CORNERS Number of box's corner
     * @param[in] NUM_OUTPUT_BOX_FEATURE Number of output box's feature
     * @details Captital variables never change after the compile, non-capital variables could be changed through rosparam
     */
    PostprocessCuda(const float FLOAT_MIN, const float FLOAT_MAX, const int NUM_ANCHOR_X_INDS,
                    const int NUM_ANCHOR_Y_INDS, const int NUM_ANCHOR_R_INDS, const int num_class,
                    const int num_head, const int Rpn_Cls_Output_Dim,
                    const float score_threshold,
                    const int NUM_THREADS, const float nms_overlap_threshold, const int NUM_BOX_CORNERS,
                    const int NUM_OUTPUT_BOX_FEATURE);

    /**
     * @brief Postprocessing for the network output
     * @param[in] rpn_box_output Box predictions from the network output
     * @param[in] rpn_cls_output Class predictions from the network output
     * @param[in] rpn_dir_output Direction predictions from the network output
     * @param[in] dev_anchor_mask Anchor mask for filtering the network output
     * @param[in] dev_anchors_px X-coordinate values for corresponding anchors
     * @param[in] dev_anchors_py Y-coordinate values for corresponding anchors
     * @param[in] dev_anchors_pz Z-coordinate values for corresponding anchors
     * @param[in] dev_anchors_dx X-dimension values for corresponding anchors
     * @param[in] dev_anchors_dy Y-dimension values for corresponding anchors
     * @param[in] dev_anchors_dz Z-dimension values for corresponding anchors
     * @param[in] dev_anchors_ro Rotation values for corresponding anchors
     * @param[in] dev_filtered_box Filtered box predictions
     * @param[in] dev_filtered_score Filtered score predictions
     * @param[in] dev_filtered_dir Filtered direction predictions
     * @param[in] dev_box_for_nms Decoded boxes in min_x min_y max_x max_y represenation from pose and dimension
     * @param[in] dev_filter_count The number of filtered output
     * @param[out] out_detection Output bounding boxes
     * @details dev_* represents device memory allocated variables
     */
    void doPostprocessCuda(const float *rpn_all_output, int *dev_anchor_mask, const float *dev_anchors_px, const float *dev_anchors_py, const float *dev_anchors_pz,
                           const float *dev_anchors_dx, const float *dev_anchors_dy, const float *dev_anchors_dz, const float *dev_anchors_ro,
                           float *dev_filtered_box, float *dev_filtered_score, int *dev_filtered_label, int *dev_filtered_dir, float *dev_box_for_nms, int *dev_filter_count,
                           std::vector<float> &out_detection, std::vector<int> &out_label, const int *num_class_each_head)
};

#endif // POSTPROCESS_CUDA_H