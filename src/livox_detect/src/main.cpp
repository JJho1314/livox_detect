// 自己定义的头文件
#include "livox_detection.hpp"

#include <ros/ros.h>

int main(int argc, char **argv)
{
    if (!build_model())
    {
        return -1;
    }
    ros::init(argc, argv, "livox_detection");

    livox_detection livox;
    livox.createROSPubSub();
    ros::spin();

    return 0;
}