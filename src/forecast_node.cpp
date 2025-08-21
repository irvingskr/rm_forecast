//
// Created by ljt666666 on 22-10-9.
//

#include "../include/forecast_node.h"
#include <pluginlib/class_list_macros.h>
#include <ros/callback_queue.h>

using namespace cv;
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

  /***线性卡尔曼滤波***/
  // Kalman Filter initial matrix
  // A - state transition matrix
  // clang-format off
  auto f = [this](const Eigen::VectorXd&) {
    Eigen::MatrixXd f(3, 3);
    // clang-format off
    f << 1,  dt_,  0.5*dt_*dt_,
      0,  1,  dt_,
      0,  0,  1;
    // clang-format on
    return f;
  };
  // clang-format on

  // H - measurement matrix
  Eigen::Matrix<double, 1, 3> h;
  h << 1, 0, 0; /***把矩阵左上角3x3赋值为对角为1其余为0***/

  // clang-format off
  auto q_v = nh.param("q", std::vector<double>{
  // angle vel accel
  0.01, 0.5, 0.1 });
  Eigen::DiagonalMatrix<double, 3> q;
  q.diagonal() << q_v[0], q_v[1], q_v[2];

  // R - measurement noise covariance matrix
  auto r_v = nh.param("r", std::vector<double>{
  // angle
  1 });
  Eigen::DiagonalMatrix<double, 1> r;
  r.diagonal() << r_v[0];
  // clang-format on

  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, 3, 3> p;
  p.setIdentity();

  kf_matrices_ = KalmanFilterMatrices{ f, h, q, r, p }; /***初始化卡尔曼滤波初始参数***/

  if (!nh.getParam("max_match_distance", max_match_distance_))
    ROS_WARN("No max match distance specified");
  if (!nh.getParam("max_match_angle", max_match_angle_))
    ROS_WARN("No max match angle specified");
  if (!nh.getParam("track_threshold", track_threshold_))
    ROS_WARN("No track threshold specified");
  if (!nh.getParam("tracking_threshold", tracking_threshold_))
    ROS_WARN("No tracking threshold specified");
  if (!nh.getParam("lost_threshold", lost_threshold_))
    ROS_WARN("No lost threshold specified");

  if (!nh.getParam("is_clockwise", is_clockwise_))
    ROS_WARN("No is_clockwise specified");
  if (!nh.getParam("fan_length", fan_length_))
    ROS_WARN("No fan_length specified");
  if (!nh.getParam("target_length", target_length_))
    ROS_WARN("No target_length specified");
  if (!nh.getParam("target_width", target_width_))
    ROS_WARN("No target_width specified");
  if (!nh.getParam("angular_velocity", angular_velocity_))
    ROS_WARN("No angular_velocity specified");
  if (!nh.getParam("delay_time", delay_time_))
    ROS_WARN("No delay_time specified");
  if (!nh.getParam("speed_threshold", speed_threshold_))
    ROS_WARN("No speed_threshold specified");
  if (!nh.getParam("high_acceleration_coefficient", high_acceleration_coefficient_))
    ROS_WARN("No high_acceleration_coefficient specified");
  if (!nh.getParam("low_acceleration_coefficient", low_acceleration_coefficient_))
    ROS_WARN("No low_acceleration_coefficient specified");
  if (!nh.getParam("high_acceleration_offset", high_acceleration_offset_))
    ROS_WARN("No high_acceleration_offset specified");
  if (!nh.getParam("low_acceleration_offset", low_acceleration_offset_))
    ROS_WARN("No low_acceleration_offset specified");
  if (!nh.getParam("skip_frame_threshold", skip_frame_threshold_))
    ROS_WARN("No skip_frame_threshold specified");
  if (!nh.getParam("is_static", is_static_))
    ROS_WARN("No is_static specified");
  if (!nh.getParam("is_small_buff", is_small_buff_))
    ROS_WARN("No is_static specified");

  forecast_cfg_srv_ = new dynamic_reconfigure::Server<rm_forecast::ForecastConfig>(ros::NodeHandle(nh_, "rm_forecast"));
  forecast_cfg_cb_ = boost::bind(&Forecast_Node::forecastconfigCB, this, _1, _2);
  forecast_cfg_srv_->setCallback(forecast_cfg_cb_);

  tracker_ = std::make_unique<Tracker>(kf_matrices_);
  tracker_->kf_.initReconfigure();

  tf_buffer_ = new tf2_ros::Buffer(ros::Duration(10));
  tf_listener_ = new tf2_ros::TransformListener(*tf_buffer_);
  points_targets_sub_ = nh.subscribe("/prediction", 1, &Forecast_Node::pointsCallback, this);
  debug_pub_ = nh.advertise<rm_msgs::TargetDetection>("/forecast/debug_result", 1);
  track_pub_ = nh.advertise<rm_msgs::TrackData>("/track", 10);

  position_marker_.ns = "position";
  position_marker_.type = visualization_msgs::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.g = 1.0;
  linear_v_marker_.type = visualization_msgs::Marker::ARROW;
  linear_v_marker_.ns = "linear_v";
  linear_v_marker_.scale.x = 0.03;
  linear_v_marker_.scale.y = 0.05;
  linear_v_marker_.color.a = 1.0;
  linear_v_marker_.color.r = 1.0;
  linear_v_marker_.color.g = 1.0;
  marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/forecast/marker", 10);

  std::vector<float> intrinsic;
  std::vector<float> distortion;
  if (!nh.getParam("/forecast/camera_matrix/data", intrinsic))
    ROS_WARN("No cam_intrinsic_mat_k specified");
  if (!nh.getParam("/forecast/distortion_coefficients/data", distortion))
    ROS_WARN("No distortion specified");

  cam_intrinsic_mat_k_ = cv::Matx<float, 3, 3>(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4],
                                               intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]);
  std::cout << "intrinsic maxtric is: " << cam_intrinsic_mat_k_ << std::endl;
  dist_coefficients_ = cv::Matx<float, 1, 5>(distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]);

  it_ = make_shared<image_transport::ImageTransport>(nh_);
  img_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 100, &Forecast_Node::imgCallback, this);
  fly_time_sub_ =
      nh.subscribe("/controllers/gimbal_controller/bullet_solver/fly_time", 10, &Forecast_Node::flyTimeCB, this);
  draw_pub_ = it_->advertise("reproject_image", 1);
}

void Forecast_Node::forecastconfigCB(rm_forecast::ForecastConfig& config, uint32_t level)
{
  if (!dynamic_reconfig_initialized_)
  {
    config.max_match_distance = max_match_distance_;
    config.tracking_threshold = tracking_threshold_;
    config.lost_threshold = lost_threshold_;

    config.is_clockwise = is_clockwise_;
    config.fan_length = fan_length_;
    config.target_length = target_length_;
    config.target_width = target_width_;
    config.angular_velocity = angular_velocity_;
    config.delay_time = delay_time_;
    config.speed_threshold = speed_threshold_;
    config.high_acceleration_coefficient = high_acceleration_coefficient_;
    config.low_acceleration_coefficient = low_acceleration_coefficient_;
    config.high_acceleration_offset = high_acceleration_offset_;
    config.low_acceleration_offset = low_acceleration_offset_;
    config.skip_frame_threshold = skip_frame_threshold_;
    config.is_static = is_static_;
    dynamic_reconfig_initialized_ = true;
  }

  max_match_distance_ = config.max_match_distance;
  tracking_threshold_ = config.tracking_threshold;
  lost_threshold_ = config.lost_threshold;

  /// reproject
  is_reproject_ = config.is_reproject;
  is_prev_reproject_ = config.is_prev_reproject;

  /// windmill_kalman
  is_clockwise_ = config.is_clockwise;
  fan_length_ = config.fan_length;
  target_length_ = config.target_length;
  target_width_ = config.target_width;
  angular_velocity_ = config.angular_velocity;
  delay_time_ = config.delay_time;
  speed_threshold_ = config.speed_threshold;
  high_acceleration_coefficient_ = config.high_acceleration_coefficient;
  low_acceleration_coefficient_ = config.low_acceleration_coefficient;
  high_acceleration_offset_ = config.high_acceleration_offset;
  low_acceleration_offset_ = config.low_acceleration_offset;
  skip_frame_threshold_ = config.skip_frame_threshold;
  is_static_ = config.is_static;
}

bool Forecast_Node::updateFan(Target& object)
{
  if (!speed_init_flag_)
  {
    prev_fan_ = object;
    speed_init_flag_ = true;
    return false;
  }
  else
  {
    last_fan_ = prev_fan_;
    prev_fan_ = object;
    return true;
  }
}

/// 计算连续的角度
void Forecast_Node::speedSolution()
{
  float angle_dif = getAngle();

  //  filter_.input(angle, prev_target.stamp);
  //  angle = filter_.output();

  frame_angle_ = angle_dif;
  angle_ += angle_dif;
}

float Forecast_Node::getAngle()
{
  cv::Point2d vec1(prev_fan_.armor_center_points.x - prev_fan_.r_points.x,
                   prev_fan_.armor_center_points.y - prev_fan_.r_points.y);
  cv::Point2d vec2(last_fan_.armor_center_points.x - last_fan_.r_points.x,
                   last_fan_.armor_center_points.y - last_fan_.r_points.y);
  cv::Point2d vec3(1, 0);
  image_fan_length_ = pow(pow(prev_fan_.armor_center_points.x - prev_fan_.r_points.x, 2) +
                              pow(prev_fan_.armor_center_points.y - prev_fan_.r_points.y, 2),
                          0.5);

  /// 判断正反方向
  if (vec1.cross(vec2) > 0)
    plus_num_++;
  else
  {
    minus_num_++;
  }
  if (plus_num_ < minus_num_)
    is_clockwise_ = true;
  else
    is_clockwise_ = false;

  /// 计算当前帧与x轴夹角以及与上一帧角度差
  auto costheta = static_cast<float>(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2)));
  auto costheta_x = static_cast<float>(vec3.dot(vec1) / (cv::norm(vec1) * cv::norm(vec3)));
  x_angle_ = acos(costheta_x);
  float angle_dif = acos(costheta);

  /// 给x轴夹角加符号（cos无法计算pi-2pi）
  if (vec3.cross(vec1) > 0)
    x_angle_ = -x_angle_;

  return angle_dif;
}

void Forecast_Node::pointsCallback(const rm_msgs::TargetDetectionArray::Ptr& msg)
{
  rm_msgs::TargetDetection debug_result;
  rm_msgs::TrackData track_data;
  track_data.header.frame_id = "odom";
  track_data.header.stamp = msg->header.stamp;
  track_data.id = 0;
  stamp_ = msg->header.stamp;

  /// R标与扇页中心
  for (auto& object : msg->detections)
  {
    float data[3 * 2];
    memcpy(&data[0], &object.pose.position.x, sizeof(float) * 2);
    memcpy(&data[2], &object.pose.position.y, sizeof(float) * 2);
    memcpy(&data[4], &object.pose.position.z, sizeof(float) * 2);
    Target target;
    target.label = object.id;

    target.r_points.x = data[0];
    target.r_points.y = data[1];
    target.armor_center_points.x = data[2];
    target.armor_center_points.y = data[3];

    if (updateFan(target))
    {
      speedSolution();
    }
  }

  /// 扇页4个点
  int32_t data[4 * 2];  // data of 4 2D points
  for (const auto& detection : msg->detections)
  {
    memcpy(&data[0], &detection.pose.orientation.x, sizeof(int32_t) * 2);
    memcpy(&data[2], &detection.pose.orientation.y, sizeof(int32_t) * 2);
    memcpy(&data[4], &detection.pose.orientation.z, sizeof(int32_t) * 2);
    memcpy(&data[6], &detection.pose.orientation.w, sizeof(int32_t) * 2);
    //    memcpy(&data[8], &detection.pose.position.x, sizeof(int32_t) * 2);
  }

  bool detect{};

  if (msg->detections.empty() || msg->detections[0].id == 0)
  {
    //    track_pub_.publish(track_data);
    //    return;
  }
  else
  {
    detect = true;
    std::vector<Point2f> pic_points = sortPoints(data);
    hit_target_ = pnp(pic_points);
    try
    {
      src_pose_.position.x = hit_target_.tvec.x();
      src_pose_.position.y = hit_target_.tvec.y();
      src_pose_.position.z = hit_target_.tvec.z();
      Eigen::Quaterniond src_quat(hit_target_.rmat);
      src_pose_.orientation.x = src_quat.x();
      src_pose_.orientation.y = src_quat.y();
      src_pose_.orientation.z = src_quat.z();
      src_pose_.orientation.w = src_quat.w();

      geometry_msgs::TransformStamped transform =
          tf_buffer_->lookupTransform("odom", "camera2_optical_frame", msg->header.stamp, ros::Duration(1));

      tf2::doTransform(src_pose_, transform_pose_, transform);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
    }
  }

  if (tracker_->tracker_state == Tracker::LOST)
  {
    angle_ = 0.1;
    tracker_->init(0.1, 0, 0, detect);
    history_info_.clear();
    //            target_msg.tracking = false;
    tracking_ = false;
    debug_result.id = tracker_->tracker_state;
    track_pub_.publish(track_data);
    debug_pub_.publish(debug_result);
    last_time_ = msg->header.stamp;
    return;
  }
  /***是其他状态则更新tracker***/
  else
  {
    // Set dt
    dt_ = (msg->header.stamp - last_time_).toSec();
    last_time_ = msg->header.stamp;

    tracker_->update(angle_, max_match_angle_, track_threshold_, lost_threshold_, detect);
    tracking_ = true;
  }

  try
  {
    geometry_msgs::Pose pose;
    geometry_msgs::TransformStamped transform =
        tf_buffer_->lookupTransform("camera2_optical_frame", "odom", msg->header.stamp, ros::Duration(1));
    tf2::doTransform(transform_pose_, pose, transform);
    hit_target_.tvec[0] = pose.position.x;
    hit_target_.tvec[1] = pose.position.y;
    hit_target_.tvec[2] = pose.position.z;
    Eigen::Quaterniond  dst_quat(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z
        );
    hit_target_.rmat = dst_quat;
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
  }

  if (tracker_->tracker_state == Tracker::TEMP_LOST)
    temp_lost_time_ = (msg->header.stamp - tlost_start_).toSec();
  else
  {
    tlost_start_ = msg->header.stamp;
    temp_lost_time_ = 0;
  }

  debug_result.id = tracker_->tracker_state;

  if (!tracking_)
    return;

  speed_ = tracker_->target_state(1);
  last_a_ = tracker_->target_state(2);

  double params[4];
  //  double a = (tracker_->target_state(3) - last_speed_) / (msg->header.stamp
  //  - last_time_).toSec();
  //  ROS_INFO("fan_length:%f", fan_length_);
  //  ROS_INFO("2d_fan_length:%f", image_fan_length_);
  //  ROS_INFO("kalman_a:%f", tracker_->target_state(4));
  //  ROS_INFO("kalman_pos:%f", tracker_->target_state(0));
  //  ROS_INFO("angle:%f", angle_);
  //  ROS_INFO("kalman_speed:%f", tracker_->target_state(3));
  //  ROS_INFO("minus:%f", (tracker_->target_state(3) - last_speed_));
  //  ROS_INFO("time:%f", (msg->header.stamp - last_time_).toSec());
  //  ROS_INFO("kalman_speed + a:%f", tracker_->target_state(3) + a);

  if (is_small_buff_)
  {
    //      params[3] = CV_PI / 3 * delay_time_;
    params[3] = angular_velocity_;
  }
  else
  {
    if (abs(speed_) > speed_threshold_)
    {
      params[3] = abs(tracker_->target_state(1)) +
                  high_acceleration_coefficient_ * (tracker_->target_state(2) + high_acceleration_offset_);
      //              params[3] = high_acceleration_coefficient_ *
      //              ((abs(last_speed_) * 2 + (last_a_ +
      //              high_acceleration_offset_) * delay_time_) * delay_time_ /
      //              2);
    }
    else
    {
      params[3] = abs(tracker_->target_state(1)) +
                  low_acceleration_coefficient_ * (tracker_->target_state(2) + low_acceleration_offset_);
      //              params[3] = low_acceleration_coefficient_ *
      //              ((abs(last_speed_) * 2 + (last_a_ +
      //              low_acceleration_offset_) * delay_time_) * delay_time_ /
      //              2);
    }

    finalTarget final_target;
    final_target.stamp = msg->header.stamp;
    final_target.speed = params[3];
    if (history_info_.size() < tracking_threshold_)
    {
      history_info_.push_back(final_target);
      is_filled_up_ = false;
      //    last_target = target;
      //    return false;
    }
    else if (history_info_.size() == tracking_threshold_)
    {
      history_info_.pop_front();
      history_info_.push_back(final_target);
      is_filled_up_ = true;
    }
    else if (history_info_.size() > tracking_threshold_)
    {
      while (history_info_.size() >= tracking_threshold_)
        history_info_.pop_front();
      history_info_.push_back(final_target);
    }
    if (is_filled_up_)
    {
      for (int i = 0; i < history_info_.size(); i++)
      {
        if (abs((history_info_[i].stamp - msg->header.stamp).toSec()) < max_match_distance_ - fly_time_)
        {
          params[3] = history_info_[i].speed;
          break;
        }
      }

      // Calculate the indefine-velocity
      vel_buf_.clear();
      for (auto& w : history_info_)
      {
        if (abs((w.stamp - msg->header.stamp).toSec()) > max_match_distance_ &&
            abs((w.stamp - msg->header.stamp).toSec()) < max_match_distance_ - fly_time_)
        {
          vel_buf_.push_back(w);
        }
      }
    }
  }

  if (params[3] < 0)
    params[3] = 0.05;

  double t0 = 0;
  double t1 = fly_time_;
  int mode = 0;
  std::vector<double> hit_point = calcAimingAngleOffset(hit_target_, params, t0, t1, mode);

  rm_msgs::TargetDetection detection_temp;
  detection_temp.pose.position.x = hit_point[0];
  detection_temp.pose.position.y = hit_point[1];
  detection_temp.pose.position.z = hit_point[2];

  geometry_msgs::PoseStamped pose_in;
  geometry_msgs::PoseStamped pose_out;
  geometry_msgs::Vector3 vec_in;
  geometry_msgs::Vector3 vec_out;
  pose_in.header.frame_id = "camera2_optical_frame";
  pose_in.header.stamp = msg->header.stamp;
  pose_in.pose = detection_temp.pose;
  if (is_clockwise_)
  {
    vec_in.x = params[3] * sin(x_angle_ - theta_offset_);
    vec_in.y = params[3] * cos(x_angle_ - theta_offset_);
  }
  else
  {
    vec_in.x = -params[3] * sin(x_angle_ + theta_offset_);
    vec_in.y = -params[3] * cos(x_angle_ + theta_offset_);
  }
  if (is_static_)
  {
    vec_in.x = 0;
    vec_in.y = 0;
  }
  vec_in.z = 0;

  debug_result.pose.position.x = tracker_->target_state(1);
  debug_result.pose.position.y = tracker_->target_state(2);
  debug_result.pose.position.z = hit_point[2];
  debug_result.pose.orientation.x = tracker_->target_state(0);
  debug_result.pose.orientation.y = params[3];
  if (frame_angle_ > 0.2 || frame_angle_ < 0)
    debug_result.pose.orientation.z = 0.05 / dt_;
  else
    debug_result.pose.orientation.z = frame_angle_ / dt_;

  debug_pub_.publish(debug_result);

  drawCallback();

  try
  {
    geometry_msgs::TransformStamped transform =
        tf_buffer_->lookupTransform("odom", pose_in.header.frame_id, msg->header.stamp, ros::Duration(1));

    tf2::doTransform(vec_in, vec_out, transform);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
  }

  try
  {
    geometry_msgs::TransformStamped transform =
        tf_buffer_->lookupTransform("odom", pose_in.header.frame_id, msg->header.stamp, ros::Duration(1));

    tf2::doTransform(pose_in.pose, pose_out.pose, transform);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
  }

  detection_temp.pose = pose_out.pose;

  track_data.id = 12;
  track_data.position.x = detection_temp.pose.position.x;
  track_data.position.y = detection_temp.pose.position.y;
  track_data.position.z = detection_temp.pose.position.z;
  track_data.velocity.x = vec_out.x;
  track_data.velocity.y = vec_out.y;
  track_data.velocity.z = vec_out.z;
  //  track_data.velocity.x = 0;
  //  track_data.velocity.y = 0;
  //  track_data.velocity.z = 0;
  track_data.armors_num = 2;
  //  track_data.target_pos.x = detection_temp.pose.position.x;
  //  track_data.target_pos.y = detection_temp.pose.position.y;
  //  track_data.target_pos.z = detection_temp.pose.position.z;
  //  track_data.target_vel.x = 0;
  //  track_data.target_vel.y = 0;
  //  track_data.target_vel.z = 0;
  track_pub_.publish(track_data);
  publishMarkers(track_data);
}

void Forecast_Node::flyTimeCB(const std_msgs::Float64ConstPtr& msg)
{
  fly_time_ = msg->data + delay_time_;
}

std::vector<cv::Point2f> Forecast_Node::sortPoints(const int32_t* data)
{
  std::vector<Point2f> pic_points;
  for (int i = 0; i < 4; i++)
  {
    Point2f point;
    point.x = float(data[2 * i]);
    point.y = float(data[2 * i + 1]);
    pic_points.emplace_back(point);
  }

  RotatedRect rect = minAreaRect(pic_points);
  Mat points;
  boxPoints(rect, points);
  std::vector<std::pair<Point2f, int>> sorted_pts;
  for (int i = 0; i < 4; ++i)
  {
    std::pair<Point2f, int> pt;
    pt.first = Point2f(points.at<float>(i, 0), points.at<float>(i, 1));
    pt.second = i;
    sorted_pts.emplace_back(pt);
  }
  std::array<Point2f, 8> pts = { sorted_pts[0].first, sorted_pts[1].first, sorted_pts[2].first, sorted_pts[3].first,
                                 sorted_pts[0].first, sorted_pts[1].first, sorted_pts[2].first, sorted_pts[3].first };
  std::sort(sorted_pts.begin(), sorted_pts.end(), [&](const auto& v1, const auto& v2) {
    return norm(v1.first - pic_points[0]) < norm(v2.first - pic_points[0]);
  });
  for (int i = 0; i < 4; ++i)
  {
    if (i == sorted_pts[0].second)
    {
      pic_points.clear();
      for (int j = 0; j < 4; ++j)
        pic_points.emplace_back(pts[i + j]);
      break;
    }
  }

  return pic_points;
}
Target Forecast_Node::pnp(const std::vector<Point2f>& points_pic)
{
  std::vector<Point3d> points_world;

  //        //长度为5进入大符模式
  //        points_world = {
  //                {-0.1125,0.027,0},
  //                {-0.1125,-0.027,0},
  ////                {0,-0.7,-0.05},
  //                {0.1125,-0.027,0},
  //                {0.1125,0.027,0}};
  //        points_world = {
  //                {-0.066,-0.027,0},
  //                {-0.066,0.027,0},
  //                {0.066,0.027,0},
  //                {0.066,-0.027,0}};
  points_world = { { -target_length_ / 2, -target_width_ / 2, 0 },
                   { -target_length_ / 2, target_width_ / 2, 0 },
                   //         {0,-0.565,-0.05},
                   { target_length_ / 2, target_width_ / 2, 0 },
                   { target_length_ / 2, -target_width_ / 2, 0 } };

  Mat rvec;
  Mat rmat;
  Mat tvec;
  Eigen::Matrix3d rmat_eigen;
  Eigen::Vector3d R_center_world = { 0, -0.7, -0.05 };
  Eigen::Vector3d tvec_eigen;
  Eigen::Vector3d coord_camera;

  solvePnP(points_world, points_pic, cam_intrinsic_mat_k_, dist_coefficients_, rvec, tvec, false, SOLVEPNP_ITERATIVE);

  std::array<double, 3> trans_vec = tvec.reshape(1, 1);
  //  ROS_INFO("x:%f, y:%f, z:%f", trans_vec[0], trans_vec[1], trans_vec[2]);

  Target result;
  //        //Pc = R * Pw + T
  cv::Rodrigues(rvec, rmat);      /***罗德里格斯变换，把旋转向量转换为旋转矩阵***/
  cv::cv2eigen(rmat, rmat_eigen); /***cv转成eigen格式***/
  cv::cv2eigen(tvec, tvec_eigen);
  //
  result.rmat = rmat_eigen;
  result.tvec = tvec_eigen;

  return result;
}

std::vector<double> Forecast_Node::calcAimingAngleOffset(Target& object, double params[4], double t0, double t1,
                                                         int mode)
{
  auto a = params[0];
  auto omega = params[1];
  auto theta = params[2];
  auto b = params[3];
  double theta1;
  double theta0;
  //  cout << "t1: " << t1 << endl;
  //  cout << "t0: " << t0 << endl;
  // f(x) = a * sin(ω * t + θ) + b
  // 对目标函数进行积分
  if (is_small_buff_)  // 适用于小符模式
  {
    theta0 = 0;
    theta1 = b;
  }
  else
  {
    theta0 = b * t0;
    theta1 = b * t1;
  }
  //  theta0 = 0;
  //  theta1 = b;
  //  cout << (theta1 - theta0) * 180 / CV_PI << endl;
  theta_offset_ = theta1 - theta0;

  if (vel_buf_.size() > 1)
  {
    theta_offset_ = 0;
    for (int i = 0; i < vel_buf_.size() - 1; i++)
    {
      theta_offset_ +=
          (vel_buf_[i].speed + vel_buf_[i + 1].speed) * abs((vel_buf_[i].stamp - vel_buf_[i + 1].stamp).toSec()) / 2;
    }
  }

  if (is_static_)
    theta_offset_ = 0;

  int clockwise_sign = is_clockwise_ == 1 ? 1 : -1;
  Eigen::Vector3d hit_point_world = { clockwise_sign * sin(theta_offset_) * fan_length_,
                                      (cos(theta_offset_) - 1) * fan_length_, 0 };
  Eigen::Vector3d hit_point_cam = (object.rmat * hit_point_world) + object.tvec;

  std::vector<double> hit_points;
  hit_points.emplace_back(hit_point_cam.transpose()[0]);
  hit_points.emplace_back(hit_point_cam.transpose()[1]);
  hit_points.emplace_back(hit_point_cam.transpose()[2]);

  target2d_ = reproject(hit_point_cam);

  if (target2d_his_.size() < tracking_threshold_)
  {
    target2d_his_.push_back(std::pair<cv::Point2f, ros::Time>(target2d_, stamp_));
  }
  else
  {
    target2d_his_.pop_front();
    target2d_his_.push_back(std::pair<cv::Point2f, ros::Time>(target2d_, stamp_));
  }

  return hit_points;
}

/**
 * @brief 重投影
 *
 * @param xyz 目标三维坐标
 * @return cv::Point2f 图像坐标系上坐标(x,y)
 */
cv::Point2f Forecast_Node::reproject(Eigen::Vector3d& xyz)
{
  Eigen::Matrix3d mat_intrinsic;
  cv2eigen(cam_intrinsic_mat_k_, mat_intrinsic);
  //(u,v,1)^T = (1/Z) * K * (X,Y,Z)^T
  auto result = (1.f / xyz[2]) * mat_intrinsic * (xyz);  // 解算前进行单位转换
  return cv::Point2f(result[0], result[1]);
}

void Forecast_Node::imgCallback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
{
  memcpy(cam_intrinsic_mat_k_.val, info->K.data(), 9 * sizeof(double));
  memcpy(dist_coefficients_.val, info->D.data(), 5 * sizeof(double));
  raw_img_ = cv_bridge::toCvShare(img, "bgr8")->image;
}

void Forecast_Node::drawCallback()
{
  if (!is_reproject_ && !is_prev_reproject_)
    return;

  cv::Mat img = raw_img_.clone();
  if (is_reproject_)
    circle(img, target2d_, 10, cv::Scalar(0, 255, 0), -1, 2);
  if (target2d_his_.size() == tracking_threshold_ && is_prev_reproject_)
  {
    for (auto& target_2d_en : target2d_his_)
    {
      if (abs((target_2d_en.second - stamp_).toSec()) < fly_time_)
      {
        circle(img, target_2d_en.first, 10, cv::Scalar(255, 255, 0), -1, 2);
        break;
      }
    }
  }
  draw_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg());
}

bool Forecast_Node::changeStatusCB(rm_msgs::StatusChange::Request& change, rm_msgs::StatusChange::Response& res)
{
  plus_num_ = 0;
  minus_num_ = 0;
  angle_ = 0;
  tracker_->tracker_state = Tracker::LOST;
  ROS_INFO("change.target is %d", change.target);
  this->is_small_buff_ = change.target == 0;

  if (is_small_buff_)
    ROS_INFO("hitting the small buff.");
  else
    ROS_INFO("hitting the big buff.");

  res.switch_is_success = true;
  return true;
}

void Forecast_Node::publishMarkers(const rm_msgs::TrackData& track_data)
{
  position_marker_.header = track_data.header;
  linear_v_marker_.header = track_data.header;
  angular_v_marker_.header = track_data.header;
  armors_marker_.header = track_data.header;

  if (tracking_)
  {
    double yaw = track_data.yaw, r1 = track_data.radius_1, r2 = track_data.radius_2;
    double xc = track_data.position.x, yc = track_data.position.y, zc = track_data.position.z;
    double dz = track_data.dz;
    position_marker_.action = visualization_msgs::Marker::ADD;
    position_marker_.pose.position.x = xc;
    position_marker_.pose.position.y = yc;
    position_marker_.pose.position.z = zc;

    linear_v_marker_.action = visualization_msgs::Marker::ADD;
    linear_v_marker_.points.clear();
    linear_v_marker_.points.emplace_back(position_marker_.pose.position);
    geometry_msgs::Point arrow_end = position_marker_.pose.position;
    arrow_end.x += track_data.velocity.x;
    arrow_end.y += track_data.velocity.y;
    arrow_end.z += track_data.velocity.z;
    linear_v_marker_.points.emplace_back(arrow_end);
  }
  else
  {
    position_marker_.action = visualization_msgs::Marker::DELETE;
    linear_v_marker_.action = visualization_msgs::Marker::DELETE;
  }

  visualization_msgs::MarkerArray marker_array;

  marker_array.markers.emplace_back(position_marker_);
  marker_array.markers.emplace_back(linear_v_marker_);
  marker_pub_.publish(marker_array);
}

}  // namespace rm_forecast
