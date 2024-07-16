//
// Created by ljt666666 on 23-2-16.
//

#ifndef RM_FORECAST_SPIN_OBSERVER_H
#define RM_FORECAST_SPIN_OBSERVER_H

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <angles/angles.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <rm_forecast/ForecastConfig.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <nodelet/nodelet.h>
#include <pluginlib/class_loader.h>
#include <rm_msgs/TargetDetection.h>
#include <rm_msgs/TargetDetectionArray.h>
#include <rm_msgs/TrackData.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <ros/ros.h>
#include <thread>
#include <vector>

using namespace std;

namespace rm_forecast {
class SpinObserver
{
public:
    SpinObserver();

    void update(rm_msgs::TrackData & target_msg, geometry_msgs::TransformStamped & odom2pitch, ros::Time &current_time, double &max_jump_angle, double &max_jump_period, double &allow_following_range);

//    auto_aim_interfaces::msg::SpinInfo spin_info_msg;
    rm_msgs::TrackData spin_track_data;

private:
    bool target_spinning_;

    double jump_period_;
    int jump_count_;

    double last_yaw_;
    double last_jump_yaw_diff_;

    ros::Time last_jump_time_;
    Eigen::Vector3d last_jump_position_;

    bool suggest_fire_;
};
} // namespace rm_forecast

#endif //RM_FORECAST_SPIN_OBSERVER_H
