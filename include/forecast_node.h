//
// Created by ljt666666 on 22-10-9.
//
#ifndef RM_FORECAST_FORECAST_H
#define RM_FORECAST_FORECAST_H

#pragma once

#include "kalman_filter.h"
#include "tracker.h"
#include "spin_observer.h"
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
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
#include <rm_msgs/TargetDetection.h>
#include <rm_msgs/TargetDetectionArray.h>
#include <rm_msgs/TrackData.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <thread>
#include <vector>

using namespace std;

namespace rm_forecast {
    struct InfoTarget
    {
        ros::Time stamp;
        float angle;
    };

    struct Target
    {
        std::vector<float> points;
        cv::Point3f armor_center_points;
        cv::Point3f r_points;
        int label;
        Eigen::Matrix3d rmat;
        Eigen::Vector3d tvec;
    };

static double max_match_distance_{};
static int tracking_threshold_{};
static int lost_threshold_{};
    static double max_jump_angle_{};
    static double max_jump_period_{};
    static double allow_following_range_{};

class Forecast_Node : public nodelet::Nodelet {
public:

  Forecast_Node() = default;

  ~Forecast_Node() override {
    if (this->my_thread_.joinable())
      my_thread_.join();
  }

  void initialize(ros::NodeHandle &nh);

  void onInit() override;

private:
  rm_msgs::TargetDetectionArray target_array_;
  ros::Subscriber targets_sub_;
  ros::Publisher track_pub_;
  ros::NodeHandle nh_;
  std::shared_ptr<image_transport::ImageTransport> it_;
  dynamic_reconfigure::Server<rm_forecast::ForecastConfig>*forecast_cfg_srv_{};
  dynamic_reconfigure::Server<rm_forecast::ForecastConfig>::CallbackType forecast_cfg_cb_;

  // transform
 //  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  tf2_ros::TransformListener *tf_listener_;
  tf2_ros::Buffer *tf_buffer_;

  // Last time received msg
  ros::Time last_time_;

  // Initial KF matrices
  KalmanFilterMatrices kf_matrices_;
  double dt_;

  // Armor tracker
  std::unique_ptr<Tracker> tracker_;
  bool tracking_{};

  //Spin observer
  std::unique_ptr<SpinObserver> spin_observer_;
  bool allow_spin_observer_ = true;

  void forecastconfigCB(rm_forecast::ForecastConfig &config, uint32_t level);

  void speedCallback(const rm_msgs::TargetDetectionArray::Ptr &msg);



    void outpostCallback(const rm_msgs::TargetDetectionArray::Ptr &msg);
  bool forecast_readied_ = false;
  bool reset_ = false;
  bool init_flag_ = false;
  bool fitting_succeeded_ = false;
  int target_quantity_ = 0;
  int min_target_quantity_ = 100;
  double line_speed_ = 0.8;
  double z_c_ = 0.5;
    double outpost_radius_ = 0.35;
    double rotate_speed_ = 0.5;
    bool circle_suggest_fire_ = false;
    double fly_time_ = 0.5;
    double y_thred_ = 0.05;
    ros::Time last_min_time_;
  double max_y_ = 0, min_y_ = 0, min_distance_ = 0;
  rm_msgs::TargetDetectionArray max_y_target_;
  rm_msgs::TargetDetectionArray min_y_target_;
    rm_msgs::TargetDetectionArray min_distance_target_;
  ros::Time min_y_time_;
  double theta_b_;

  std::thread my_thread_;
  image_transport::Publisher image_pub_;

  ///draw reproject
    void drawCallback(const sensor_msgs::ImageConstPtr& img);

    template<typename T>
    bool initMatrix(Eigen::MatrixXd &matrix,std::vector<T> &vector);

    ros::Subscriber draw_sub_;
    image_transport::Publisher draw_pub_;

    bool is_reproject_;
    double re_fly_time_;
    cv::Mat m_intrinsic_;

    ///windmill kalman
    void pointsCallback(const rm_msgs::TargetDetectionArray::Ptr &msg);

    Target pnp(const std::vector<cv::Point2f>& points_pic);

    std::vector<double> calcAimingAngleOffset(Target& object, double params[4], double t0, double t1, int mode);

    cv::Point2f reproject(Eigen::Vector3d &xyz);

    cv::Matx33f cam_intrinsic_mat_k_;
//    std::vector<double> dist_coefficients_;
    cv::Matx<float, 1, 5> dist_coefficients_;

    const double fan_length_ = 0.7; //大符臂长(R字中心至装甲板中心)
    double delay_time_ = 0.5;
    cv::Point2f target2d_{};
    cv::Point2f r2d_{};
    cv::Point2f last_target2d_{};
    cv::Point2f last_r2d_{};
};

} // namespace rm_forecast

#endif // RM_FORECAST_FORECAST_H