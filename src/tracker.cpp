//
// Created by ljt666666 on 22-10-9.
//
#include "../include/forecast_node.h"
#include "../include/tracker.h"

namespace rm_forecast
{
    Tracker::Tracker(const KalmanFilterMatrices & kf_matrices)
            : tracker_state(LOST),
              kf_matrices_(kf_matrices),
              tracking_velocity_(Eigen::Vector3d::Zero())
    {
      kf_ = *new KalmanFilter(kf_matrices_);
      ros::NodeHandle nh;
      debug_pub_ = nh.advertise<rm_msgs::TargetDetection>("/tracker/debug_result", 1);
    }

    void Tracker::init(const float &angle, const float &speed,
                       const float &q_speed, bool detect) {
      if (detect)
      {
        // KF init
        Eigen::VectorXd init_state(3);
        //        const auto position = chosen_armor.pose.position;
        //        init_state << position.x, position.y, position.z, 0, 0, 0;
        init_state << angle, speed, q_speed;
        kf_.init(init_state);

        tracker_state = DETECTING; /***设置装甲板状态为目标识别中，需要更多帧识别到才开始跟踪***/
      }
      else
        tracker_state = LOST;
    }

    void Tracker::update(const double &angle,
                         const double &max_match_distance,
                         const int &tracking_threshold,
                         const int &lost_threshold, bool detect) {

      rm_msgs::TargetDetection debug_result;

      // KF predict
      target_state = kf_.predict(); /***得出本时刻预测值***/

      debug_result.pose.position.x = angle;
      debug_result.pose.position.y = target_state(0);
      debug_result.pose.position.z = abs(angle - target_state(0));

      if (abs(angle - target_state(0)) > max_match_distance || isnan(angle))
      {
        debug_result.id = 0;
        /// 扇页跳变
        target_state(0) = angle;
        kf_.setState(target_state);
        detect = false;
      }
      else
      {
        debug_result.id = 1;
        Eigen::VectorXd position_vec(1);
        position_vec << angle;
        /***只有这个时候才做卡尔曼滤波的更新部分***/
        target_state = kf_.update(position_vec);
        /***如果离预测目标最近的装甲板的距离大于阈值，则寻找另一个id相同的装甲板重新作为追踪目标，并把上一个追踪目标的速度给它***/
      }

      debug_result.pose.orientation.x = angle;
      debug_result.pose.orientation.y = target_state(0);

      debug_pub_.publish(debug_result);
      // Tracking state machine
      if (tracker_state == DETECTING)
      {
        if (detect)
        {
          detect_count_++;
          if (detect_count_ > tracking_threshold)
          {
            detect_count_ = 0;
            tracker_state = TRACKING;
          }
        }
        else
        {
          detect_count_ = 0;
          tracker_state = LOST;
        }
      }
      else if (tracker_state == TRACKING)
      {
        if (!detect)
        {
          tracker_state = TEMP_LOST;
          lost_count_++;
        }
      }
      else if (tracker_state == TEMP_LOST)
      {
        if (!detect)
        {
          lost_count_++;
          if (lost_count_ > lost_threshold)
          {
            lost_count_ = 0;
            tracker_state = LOST;
          }
        }
        else
        {
          tracker_state = TRACKING;
          lost_count_ = 0;
        }
      }
    }

}  // namespace rm_auto_aim