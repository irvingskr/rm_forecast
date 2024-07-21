//
// Created by ljt666666 on 22-10-9.
//

#pragma once

#include "kalman_filter.h"
#include "rm_common/linear_interpolation.h"
#include <rm_common/filters/filters.h>
#include "rm_common/ori_tool.h"
#include <rm_common/tf_rt_broadcaster.h>
#include "rm_common/ros_utilities.h"
#include "spin_observer.h"
#include "tracker.h"
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <XmlRpcValue.h>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <mutex>
#include <nodelet/nodelet.h>
#include <pluginlib/class_loader.h>
#include <rm_forecast/ForecastConfig.h>
#include <rm_msgs/StatusChange.h>
#include <rm_msgs/TargetDetection.h>
#include <rm_msgs/TargetDetectionArray.h>
#include <rm_msgs/TrackData.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <thread>
#include <vector>
#include <realtime_tools/realtime_buffer.h>

using namespace std;
namespace rm_forecast
{
static double max_match_distance_{};
static int tracking_threshold_{};
static int lost_threshold_{};
static double max_jump_angle_{};
static double max_jump_period_{};
static double allow_following_range_{};

struct Config
{
  double const_distance, outpost_radius, rotate_speed, y_thred, time_thred, time_offset, ramp_time_offset,
      ramp_threshold, ring_highland_distance_offset, source_island_distance_offset, next_thred;
  int min_target_quantity;
  bool forecast_readied, reset;
};

class Forecast_Node : public nodelet::Nodelet
{
public:
  Forecast_Node() : detection_filter_(100), track_filter_(20){};
  ~Forecast_Node() override
  {
    if (this->my_thread_.joinable())
      my_thread_.join();
  }
  void initialize(ros::NodeHandle& nh);
  void onInit() override;

private:
  rm_msgs::TargetDetectionArray target_array_;
  ros::Subscriber enemy_targets_sub_;
  ros::Subscriber outpost_targets_sub_;
  ros::Subscriber fly_time_sub_;
  ros::Publisher track_pub_, suggest_fire_pub_;
  ros::NodeHandle nh_;
  std::shared_ptr<image_transport::ImageTransport> it_;
  dynamic_reconfigure::Server<rm_forecast::ForecastConfig>* forecast_cfg_srv_{};
  dynamic_reconfigure::Server<rm_forecast::ForecastConfig>::CallbackType forecast_cfg_cb_;

  // transform
  tf2_ros::TransformListener* tf_listener_;
  tf2_ros::Buffer* tf_buffer_;

  // Last time received msg
  ros::Time last_time_;
  ros::Time last_min_time_;
  ros::Time temp_min_time_;

  // Initial KF matrices
  KalmanFilterMatrices kf_matrices_;
  double dt_;

  // Armor tracker
  std::unique_ptr<Tracker> tracker_;
  bool tracking_{};

  // Spin observer
  std::unique_ptr<SpinObserver> spin_observer_;
  bool allow_spin_observer_ = false;

  void forecastconfigCB(rm_forecast::ForecastConfig& config, uint32_t level);
  void speedCallback(const rm_msgs::TargetDetectionArray::Ptr& msg);
  void outpostCallback(const rm_msgs::TargetDetectionArray::Ptr& msg);
  void flyTimeCB(const std_msgs::Float64ConstPtr& msg);
  bool changeStatusCB(rm_msgs::StatusChange::Request& change, rm_msgs::StatusChange::Response& res);
  geometry_msgs::Pose computeCircleCenter(const rm_msgs::TargetDetection point_1,
                                          const rm_msgs::TargetDetection point_2,
                                          const rm_msgs::TargetDetection point_3,
                                          const rm_msgs::TargetDetection point_4);

  bool ramp_flag_{}, outpost_flag_{}, base_flag_{};
  int armor_type_ = 0, target_quantity_ = 0;
  double min_distance_x_{}, min_distance_y_{}, min_distance_z_{}, temp_min_distance_x_{}, temp_min_distance_y_{},
      temp_min_distance_z_{};
  double fly_time_{}, bullet_solver_fly_time_{}, pitch_enter_time_{}, last_pitch_time_{};

  rm_msgs::TargetDetectionArray max_x_target_;
  rm_msgs::TargetDetectionArray min_x_target_;
  rm_msgs::TargetDetectionArray min_distance_target_;

  double min_camera_distance_;
  rm_common::LinearInterp interpolation_fly_time_;
  rm_common::LinearInterp interpolation_base_distance_on_ring_highland_;
  rm_common::LinearInterp interpolation_base_distance_on_resource_island_;
  rm_common::TfRtBroadcaster tf_broadcaster_;

  Vector3WithFilter<double> track_filter_;
  MovingAverageFilter<double> detection_filter_;

  ros::ServiceServer status_change_srv_;

  std::thread my_thread_;
  image_transport::Publisher image_pub_;
  ros::Publisher duration_pub_;

  Config config_{};
  realtime_tools::RealtimeBuffer<Config> config_rt_buffer_;
  bool dynamic_reconfig_initialized_ = false;

  bool get_target_ = false;

  geometry_msgs::Point target_;
};
}  // namespace rm_forecast
