//
// Created by ljt666666 on 22-10-9.
//

#include "../include/forecast_node.h"
#include <pluginlib/class_list_macros.h>
#include <ros/callback_queue.h>

using namespace std;

PLUGINLIB_EXPORT_CLASS(rm_forecast::Forecast_Node, nodelet::Nodelet)

namespace rm_forecast
{
void Forecast_Node::onInit()
{
  ros::NodeHandle& nh = getMTPrivateNodeHandle();
  this->status_change_srv_ = nh.advertiseService("status_switch", &Forecast_Node::changeStatusCB, this);
  static ros::CallbackQueue my_queue;
  nh.setCallbackQueue(&my_queue);
  initialize(nh);
  my_thread_ = std::thread([]() {
    ros::SingleThreadedSpinner spinner;
    spinner.spin(&my_queue);
  });
}

void Forecast_Node::initialize(ros::NodeHandle& nh)
{
  nh_ = ros::NodeHandle(nh, "rm_forecast");
  it_ = make_shared<image_transport::ImageTransport>(nh_);

  ROS_INFO("starting ProcessorNode!");

  // Kalman Filter initial matrix
  // A - state transition matrix
  // clang-format off
        Eigen::Matrix<double, 6, 6> f;
        f <<  1,  0,  0, dt_, 0,  0,
                0,  1,  0,  0, dt_, 0,
                0,  0,  1,  0,  0, dt_,
                0,  0,  0,  1,  0,  0,
                0,  0,  0,  0,  1,  0,
                0,  0,  0,  0,  0,  1;
  // clang-format on

  // H - measurement matrix
  Eigen::Matrix<double, 3, 6> h;
  h.setIdentity(); /***把矩阵左上角3x3赋值为对角为1其余为0***/

  // Q - process noise covariance matrix
  Eigen::DiagonalMatrix<double, 6> q;
  q.diagonal() << 700, 700, 700, 1400, 1400, 1400;

  // R - measurement noise covariance matrix
  Eigen::DiagonalMatrix<double, 3> r;
  r.diagonal() << 0.05, 0.05, 0.05;

  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, 6> p;
  p.setIdentity();

  kf_matrices_ = KalmanFilterMatrices{ f, h, q, r, p }; /***初始化卡尔曼滤波初始参数***/

  config_ = { .const_distance = getParam(nh, "const_distance", 0.),
              .outpost_radius = getParam(nh, "outpost_radius", 0.1),
              .rotate_speed = getParam(nh, "rotate_speed", 0.1),
              .y_thred = getParam(nh, "y_thred", 0.),
              .time_thred = getParam(nh, "time_thred", 0.),
              .time_offset = getParam(nh, "time_offset", 0.),
              .ramp_time_offset = getParam(nh, "ramp_time_offset", 0.),
              .ramp_threshold = getParam(nh, "ramp_threshold", 0.),
              .ring_highland_distance_offset = getParam(nh, "ring_highland_distance_offset", 0.),
              .source_island_distance_offset = getParam(nh, "source_island_distance_offset", 0.),
              .min_target_quantity = getParam(nh, "min_target_quantity", 0),
              .forecast_readied = getParam(nh, "forecast_readied", true),
              .reset = getParam(nh, "reset", false) };
  config_rt_buffer_.initRT(config_);

  XmlRpc::XmlRpcValue xml_rpc_value1;
  if (!nh.getParam("interpolation_fly_time", xml_rpc_value1))
    ROS_ERROR("Fly time no defined (namespace: %s)", nh.getNamespace().c_str());
  else
    interpolation_fly_time_.init(xml_rpc_value1);

  XmlRpc::XmlRpcValue xml_rpc_value2;
  if (!nh.getParam("interpolation_base_distance_on_ring_highland", xml_rpc_value2))
    ROS_ERROR("Base distance on ring highland no defined (namespace: %s)", nh.getNamespace().c_str());
  else
    interpolation_base_distance_on_ring_highland_.init(xml_rpc_value2);

  XmlRpc::XmlRpcValue xml_rpc_value3;
  if (!nh.getParam("interpolation_base_distance_on_resource_island", xml_rpc_value3))
    ROS_ERROR("Base distance resource island no defined (namespace: %s)", nh.getNamespace().c_str());
  else
    interpolation_base_distance_on_resource_island_.init(xml_rpc_value3);

  tracker_ = std::make_unique<Tracker>(kf_matrices_);

  spin_observer_ = std::make_unique<SpinObserver>();

  forecast_cfg_srv_ = new dynamic_reconfigure::Server<rm_forecast::ForecastConfig>(ros::NodeHandle(nh_, "rm_forecast"));
  forecast_cfg_cb_ = boost::bind(&Forecast_Node::forecastconfigCB, this, _1, _2);
  forecast_cfg_srv_->setCallback(forecast_cfg_cb_);

  tf_buffer_ = new tf2_ros::Buffer(ros::Duration(10));
  tf_listener_ = new tf2_ros::TransformListener(*tf_buffer_);
  tf_broadcaster_.init(nh);
  ROS_INFO("armor type is : %d", armor_type_);

  //  enemy_targets_sub_ = nh.subscribe("/detection", 1, &Forecast_Node::speedCallback, this);
  outpost_targets_sub_ = nh.subscribe("/outpost_detection", 1, &Forecast_Node::outpostCallback, this);
  track_pub_ = nh.advertise<rm_msgs::TrackData>("/track", 10);
  suggest_fire_pub_ = nh.advertise<std_msgs::Bool>("suggest_fire", 1);
  duration_pub_ = nh.advertise<std_msgs::Float64>("duration", 1);
  fly_time_sub_ =
      nh.subscribe("/controllers/gimbal_controller/bullet_solver/fly_time", 10, &Forecast_Node::flyTimeCB, this);
}

void Forecast_Node::forecastconfigCB(rm_forecast::ForecastConfig& config, uint32_t level)
{
  ROS_INFO("[Forecast] Dynamic params change");
  if (!dynamic_reconfig_initialized_)
  {
    Config init_config = *config_rt_buffer_.readFromNonRT();  // config init use yaml
    config.const_distance = init_config.const_distance;
    config.outpost_radius = init_config.outpost_radius;
    config.rotate_speed = init_config.rotate_speed;
    config.time_thred = init_config.time_thred;
    config.time_offset = init_config.time_offset;
    config.ramp_time_offset = init_config.ramp_time_offset;
    config.ramp_threshold = init_config.ramp_threshold;
    config.min_target_quantity = init_config.min_target_quantity;
    config.forecast_readied = init_config.forecast_readied;
    config.reset = init_config.reset;
    config.y_thred = init_config.y_thred;
    config.ring_highland_distance_offset = init_config.ring_highland_distance_offset;
    config.source_island_distance_offset = init_config.source_island_distance_offset;
    dynamic_reconfig_initialized_ = true;
  }
  Config config_non_rt{
    .const_distance = config.const_distance,
    .outpost_radius = config.outpost_radius,
    .rotate_speed = config.rotate_speed,
    .y_thred = config.y_thred,
    .time_thred = config.time_thred,
    .time_offset = config.time_offset,
    .ramp_time_offset = config.ramp_time_offset,
    .ramp_threshold = config.ramp_threshold,
    .ring_highland_distance_offset = config.ring_highland_distance_offset,
    .source_island_distance_offset = config.source_island_distance_offset,
    .next_thred = config.next_thred,
    .min_target_quantity = config.min_target_quantity,
    .forecast_readied = config.forecast_readied,
    .reset = config.reset,
  };
  config_rt_buffer_.writeFromNonRT(config_non_rt);
}

void Forecast_Node::outpostCallback(const rm_msgs::TargetDetectionArray::Ptr& msg)
{
  config_ = *config_rt_buffer_.readFromRT();

  // Initialize track data
  rm_msgs::TrackData track_data;
  track_data.header.frame_id = "odom";
  track_data.header.stamp = target_array_.header.stamp;
  track_data.id = 0;

  double duration = (ros::Time::now() - last_min_time_).toSec();
  std_msgs::Float64 duration_msg;
  duration_msg.data = duration;
  duration_pub_.publish(duration_msg);
  std_msgs::Bool circle_suggest_fire;
  circle_suggest_fire.data = false;

  if (abs(fly_time_ - duration) < config_.time_thred)
  {
    circle_suggest_fire.data = true;
  }
  suggest_fire_pub_.publish(circle_suggest_fire);

  // No target
  if (msg->detections.empty())
  {
    track_pub_.publish(track_data);
    return;
  }

  // Tranform armor position from image frame to world coordinate
  this->target_array_.detections.clear();
  target_array_.header = msg->header;
  bool outpost_flag = false;
  for (const auto& detection : msg->detections)
  {
    if (detection.id == 6)
    {
      outpost_flag = true;
    }
    else
    {
      continue;
    }

    double base_roll;
    geometry_msgs::TransformStamped odom2base;
    try
    {
      double pitch, yaw;
      quatToRPY(tf_buffer_->lookupTransform("odom", "base_link", msg->header.stamp, ros::Duration(1)).transform.rotation,
                base_roll, pitch, yaw);
      ROS_INFO_STREAM("base_roll" << base_roll);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
    }

    /*判断斜坡*/
    if (std::abs(base_roll) > config_.ramp_threshold)
    {
      ramp_flag_ = true;
      ROS_INFO_STREAM("In ramp");
    }
    else
    {
      ramp_flag_ = false;
      ROS_INFO_STREAM("In flat");
    }

    rm_msgs::TargetDetection detection_temp;
    geometry_msgs::PoseStamped pose_in, pose_out;
    if (ramp_flag_)
      pose_in.header.frame_id = "camera2_optical_frame";
    else
      pose_in.header.frame_id = msg->header.frame_id;
    pose_in.header.stamp = msg->header.stamp;
    pose_in.pose = detection.pose;
    try
    {
      ROS_INFO_STREAM("frame_id" << pose_in.header.frame_id);
      geometry_msgs::TransformStamped transform =
          tf_buffer_->lookupTransform("odom", pose_in.header.frame_id, msg->header.stamp, ros::Duration(1));
      tf2::doTransform(pose_in.pose, pose_out.pose, transform);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
    }

    detection_temp.id = detection.id;
    detection_temp.confidence = detection.confidence;
    detection_temp.pose = pose_out.pose;
    target_array_.detections.emplace_back(detection_temp);

    std::cout << "base_roll is " << base_roll << std::endl;
  }
  /*识别id不为前哨站*/
  if (!outpost_flag)
    return;

  if (config_.forecast_readied)
  {
    geometry_msgs::Quaternion qua2;
    qua2.x = msg->detections[0].pose.orientation.x;
    qua2.y = msg->detections[0].pose.orientation.y;
    qua2.z = msg->detections[0].pose.orientation.z;
    qua2.w = msg->detections[0].pose.orientation.w;

    double roll, pitch, yaw;
    quatToRPY(qua2, roll, pitch, yaw);

    /*判断是否正对*/
    if (std::abs(pitch) < config_.y_thred)
    {
      pitch_enter_time_ = ros::Time::now().toSec();
      /*正对并且与上次正对时间差0.3*/
      if (pitch_enter_time_ - last_pitch_time_ > config_.next_thred)
      {
        target_quantity_ = 0;
      }

      /*上述条件都满足则记录为下一块正对装甲版*/
      if (target_quantity_ == 0)
      {
        last_pitch_time_ = pitch_enter_time_;
        min_camera_distance_ = msg->detections[0].pose.position.z;
      }

      ROS_INFO("test");

      /*在正对角度阈值内不断更新正对xyz缓存*/
      if (ros::Time::now().toSec() - last_pitch_time_ < 0.2 && msg->detections[0].pose.position.z < min_camera_distance_)
      {
        temp_min_distance_x_ = target_array_.detections[0].pose.position.x;
        temp_min_distance_y_ = target_array_.detections[0].pose.position.y;
        temp_min_distance_z_ = target_array_.detections[0].pose.position.z;
        min_camera_distance_ = msg->detections[0].pose.position.z;
        temp_min_time_ = ros::Time::now();
      }

      /*在同一块装甲版的正对时间大于一定阈值则认为找到正对时的xyz*/
      if (target_quantity_ > config_.min_target_quantity)
      {
        last_min_time_ = temp_min_time_;
        min_distance_x_ = temp_min_distance_x_;
        min_distance_y_ = temp_min_distance_y_;
        min_distance_z_ = temp_min_distance_z_;
        get_target_ = true;
      }

      target_quantity_++;
    }

    /*计算等待时间*/
    if (ramp_flag_)
    {
      /*两个正对状态相差时间 - 飞行时间 - 发蛋延迟*/
      fly_time_ = (config_.ramp_time_offset - bullet_solver_fly_time_ - 0.119);
      ROS_INFO_STREAM("In ramp");
    }
    else
    {
      fly_time_ = (config_.time_offset - bullet_solver_fly_time_ - 0.119);
      ROS_INFO_STREAM("In flat");
    }
    ROS_INFO("test2");

    track_data.id = target_array_.detections[0].id;
    track_data.header.frame_id = "odom";
    if (!get_target_)
    {
      track_data.position.x = target_array_.detections[0].pose.position.x;
      track_data.position.y = target_array_.detections[0].pose.position.y;
      track_data.position.z = target_array_.detections[0].pose.position.z;
    }
    else
    {
      track_data.position.x = min_distance_x_;
      track_data.position.y = min_distance_y_;
      track_data.position.z = min_distance_z_;
    }
    track_data.velocity.x = 0;
    track_data.velocity.y = 0;
    track_data.velocity.z = 0;
    track_data.armors_num = 2;
  }
  else
  {
    track_data.id = target_array_.detections[0].id;
    track_data.position.x = target_array_.detections[0].pose.position.x;
    track_data.position.y = target_array_.detections[0].pose.position.y;
    track_data.position.z = target_array_.detections[0].pose.position.z;
    track_data.velocity.x = 0;
    track_data.velocity.y = 0;
    track_data.velocity.z = 0;
  }

  track_pub_.publish(track_data);
}

void Forecast_Node::speedCallback(const rm_msgs::TargetDetectionArray::Ptr& msg)
{
  if (armor_type_)
    return;

  config_ = *config_rt_buffer_.readFromRT();

  rm_msgs::TrackData track_data;
  track_data.header.stamp = msg->header.stamp;

  if (msg->detections.empty())
  {
    track_data.id = 0;
    track_pub_.publish(track_data);
    return;
  }

  string target_link;
  if (msg->header.frame_id == "camera2_optical_frame")
  {
    target_link = "yaw";
  }
  else
  {
    target_link = "odom";
  }

  // Tranform armor position from image frame to world coordinate
  this->target_array_.detections.clear();
  target_array_.header = msg->header;
  for (const auto& detection : msg->detections)
  {
    rm_msgs::TargetDetection detection_temp;
    geometry_msgs::PoseStamped pose_in;
    geometry_msgs::PoseStamped pose_out;
    pose_in.header.frame_id = msg->header.frame_id;
    pose_in.header.stamp = msg->header.stamp;
    pose_in.pose = detection.pose;
    try
    {
      geometry_msgs::TransformStamped transform =
          tf_buffer_->lookupTransform(target_link, pose_in.header.frame_id, msg->header.stamp, ros::Duration(1));
      tf2::doTransform(pose_in.pose, pose_out.pose, transform);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
    }

    detection_temp.id = detection.id;
    detection_temp.confidence = detection.confidence;
    detection_temp.pose = pose_out.pose;
    target_array_.detections.emplace_back(detection_temp);
  }

  /***如果是丢失状态则初始化tracker***/
  if (tracker_->tracker_state == Tracker::LOST)
  {
    tracker_->init(target_array_);
    tracking_ = false;
  }
  /***是其他状态则更新tracker***/
  else
  {
    // Set dt
    dt_ = (msg->header.stamp - last_time_).toSec();
    // Update state
    tracker_->update(target_array_, dt_, max_match_distance_, tracking_threshold_, lost_threshold_);
    /***判断是否处于跟踪状态 ***/
    if (tracker_->tracker_state == Tracker::DETECTING)
    {
      tracking_ = false;
    }
    else if (tracker_->tracker_state == Tracker::TRACKING || tracker_->tracker_state == Tracker::TEMP_LOST)
    {
      tracking_ = true;
    }
  }
  last_time_ = msg->header.stamp;

  track_data.header.frame_id = "odom";
  track_data.header.stamp = msg->header.stamp;
  track_data.id = tracker_->tracking_id;
  if (msg->header.frame_id == "camera2_optical_frame")
  {
    track_data.header.frame_id = "odom";
    double track_pos[3]{ target_array_.detections[0].pose.position.x, target_array_.detections[0].pose.position.y,
                         target_array_.detections[0].pose.position.z };
    track_filter_.input(track_pos);
    detection_filter_.input(msg->detections[0].pose.position.z);

    //    track_data.position.x = track_filter_.x();
    //    track_data.position.z = track_filter_.z();
    //    track_data.position.y = track_filter_.y();

    //    track_data.position.x = target_array_.detections[0].pose.position.x;
    //    track_data.position.z = target_array_.detections[0].pose.position.z;
    //    track_data.position.y = target_array_.detections[0].pose.position.y;
    geometry_msgs::TransformStamped odom2yaw;
    geometry_msgs::TransformStamped odom2virtual;
    tf2::Quaternion q;
    double yaw{};
    try
    {
      double roll, pitch;
      odom2yaw = tf_buffer_->lookupTransform("odom", "yaw", msg->header.stamp);
      quatToRPY(odom2yaw.transform.rotation, roll, pitch, yaw);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
    }
    q.setRPY(0., 0., yaw);

    odom2virtual.header.frame_id = "odom";
    odom2virtual.child_frame_id = "virtual_frame";
    odom2virtual.header.stamp = msg->header.stamp;
    odom2virtual.transform.translation = odom2yaw.transform.translation;
    odom2virtual.transform.rotation.x = q.x();
    odom2virtual.transform.rotation.y = q.y();
    odom2virtual.transform.rotation.z = q.z();
    odom2virtual.transform.rotation.w = q.w();

    tf_buffer_->setTransform(odom2virtual, "rm_forecast");
    //    tf_broadcaster_.sendTransform(odom2virtual);

    target_.y = target_array_.detections[0].pose.position.y;
    //    target_.x = config_.const_distance + config_.ring_highland_distance_offset;
    //    target_.z = 0.71;

    if (msg->detections[0].pose.position.z > 8.3)
    {
      target_.x = interpolation_base_distance_on_resource_island_.output(detection_filter_.output()) +
                  config_.source_island_distance_offset;
      target_.z = 0.71;
    }
    else
    {
      target_.x = interpolation_base_distance_on_ring_highland_.output(detection_filter_.output()) +
                  +config_.ring_highland_distance_offset;
      target_.z = 0.14;
    }
    tf2::doTransform(target_, track_data.position, odom2virtual);
    track_data.velocity.x = 0;
    track_data.velocity.y = 0;
    track_data.velocity.z = 0;
  }
  else if (tracking_)
  {
    //    track_data.position.x = tracker_->target_state(0);
    //    track_data.position.y = tracker_->target_state(1);
    //    track_data.position.z = tracker_->target_state(2);
    //    track_data.velocity.x = tracker_->target_state(3);
    //    track_data.velocity.y = tracker_->target_state(4);
    //    track_data.velocity.z = tracker_->target_state(5);
    track_data.position.x = tracker_->target_state(0);
    track_data.position.y = tracker_->target_state(1);
    track_data.position.z = tracker_->target_state(2);
    track_data.velocity.x = tracker_->target_state(3);
    track_data.velocity.y = tracker_->target_state(4);
    track_data.velocity.z = tracker_->target_state(5);
    track_data.armors_num = 2;
    //    track_data.velocity.x = 0;
    //    track_data.velocity.y = 0;
    //    track_data.velocity.z = 0;

    geometry_msgs::TransformStamped odom2pitch = tf_buffer_->lookupTransform("odom", "pitch", msg->header.stamp);
    /***根据观察旋转的结果决定是否建议开火***/
    if (allow_spin_observer_ && spin_observer_)
    {
      //        spin_observer_->max_jump_angle =
      //        get_parameter("spin_observer.max_jump_angle").as_double();
      //        spin_observer_->max_jump_period =
      //        get_parameter("spin_observer.max_jump_period").as_double();
      //        spin_observer_->allow_following_range =
      //        get_parameter("spin_observer.allow_following_range").as_double();

      //      spin_observer_->update(track_data, odom2pitch, msg->header.stamp, max_jump_angle_, max_jump_period_,
      //                             allow_following_range_);
      //        spin_info_pub_->publish(spin_observer_->spin_info_msg);
    }
  }

  geometry_msgs::TransformStamped transform;
  transform.header.frame_id = "odom";
  transform.header.stamp = msg->header.stamp;
  transform.child_frame_id = "target_link";
  transform.transform.translation.x = track_data.position.x;
  transform.transform.translation.y = track_data.position.y;
  transform.transform.translation.z = track_data.position.z;
  transform.transform.rotation.x = 0.;
  transform.transform.rotation.y = 0.;
  transform.transform.rotation.z = 0.;
  transform.transform.rotation.w = 1.;
  tf_broadcaster_.sendTransform(transform);

  track_pub_.publish(track_data);
}

geometry_msgs::Pose Forecast_Node::computeCircleCenter(const rm_msgs::TargetDetection point_1,
                                                       const rm_msgs::TargetDetection point_2,
                                                       const rm_msgs::TargetDetection point_3,
                                                       const rm_msgs::TargetDetection point_4)
{
  std::vector<double> p1 = { point_1.pose.position.x, point_1.pose.position.y, point_1.pose.position.z };
  std::vector<double> p2 = { point_2.pose.position.x, point_2.pose.position.y, point_2.pose.position.z };
  std::vector<double> p3 = { point_3.pose.position.x, point_3.pose.position.y, point_3.pose.position.z };
  std::vector<double> p4 = { point_4.pose.position.x, point_4.pose.position.y, point_4.pose.position.z };

  geometry_msgs::Pose center{};
  double a = p1[0] - p2[0], b = p1[1] - p2[1], c = p1[2] - p2[2];
  double a1 = p3[0] - p4[0], b1 = p3[1] - p4[1], c1 = p3[2] - p3[2];
  double a2 = p2[0] - p3[0], b2 = p2[1] - p3[1], c2 = p2[2] - p3[2];
  double D = a * b1 * c2 + a2 * b * c1 + c * a1 * b2 - (a2 * b1 * c + a1 * b * c2 + a * b2 * c1);

  if (D == 0)
  {
    return center;
  }

  double A = p1[0] * p1[0] - p2[0] * p2[0];
  double B = p1[1] * p1[1] - p2[1] * p2[1];
  double C = p1[2] * p1[2] - p2[2] * p2[2];
  double A1 = p3[0] * p3[0] - p4[0] * p4[0];
  double B1 = p3[1] * p3[1] - p4[1] * p4[1];
  double C1 = p3[2] * p3[2] - p4[2] * p4[2];
  double A2 = p2[0] * p2[0] - p3[0] * p3[0];
  double B2 = p2[1] * p2[1] - p3[1] * p3[1];
  double C2 = p2[2] * p2[2] - p3[2] * p3[2];
  double P = (A + B + C) / 2;
  double Q = (A1 + B1 + C1) / 2;
  double R = (A2 + B2 + C2) / 2;

  double Dx = P * b1 * c2 + b * c1 * R + c * Q * b2 - (c * b1 * R + P * c1 * b2 + Q * b * c2);
  double Dy = a * Q * c2 + P * c1 * a2 + c * a1 * R - (c * Q * a2 + a * c1 * R + c2 * P * a1);
  double Dz = a * b1 * R + b * Q * a2 + P * a1 * b2 - (a2 * b1 * P + a * Q * b2 + R * b * a1);

  center.position.x = Dx / D;
  center.position.y = Dy / D;
  center.position.z = Dz / D;

  return center;
}

void Forecast_Node::flyTimeCB(const std_msgs::Float64ConstPtr& msg)
{
  bullet_solver_fly_time_ = msg->data;
}

bool Forecast_Node::changeStatusCB(rm_msgs::StatusChange::Request& change, rm_msgs::StatusChange::Response& res)
{
  this->armor_type_ = change.armor_target == 1;
  //->use_id_cls_ = change.use_id_classification;
  ROS_INFO("armor type is : %d", armor_type_);
  res.switch_is_success = true;
  return true;
}

}  // namespace rm_forecast
