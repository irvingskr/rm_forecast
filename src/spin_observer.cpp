//
// Created by ljt666666 on 23-2-16.
//

#include "../include/spin_observer.h"

namespace rm_forecast
{
SpinObserver::SpinObserver()
{
  target_spinning_ = false;
  jump_period_ = 0.0;
  jump_count_ = 0;
  last_jump_time_ = ros::Time::now();
  last_jump_position_ = Eigen::Vector3d(0, 0, 0);
}

void SpinObserver::update(rm_msgs::TrackData& target_msg, geometry_msgs::TransformStamped& odom2pitch,
                          ros::Time& current_time, double& max_jump_angle, double& max_jump_period,
                          double& allow_following_range)
{
  //        ros::Time current_time = target_msg.header.stamp;

  //      Eigen::Vector3d current_position(target_msg.target_pos.x,
  //                                       target_msg.target_pos.y,
  //                                       target_msg.target_pos.z);
  //
  //      double time_after_jumping =
  //          ((current_time).toSec() - (last_jump_time_).toSec());
  //
  //      if (time_after_jumping > max_jump_period) {
  //        target_spinning_ = false;
  //        jump_count_ = 0;
  //      }
  //
  //      double current_yaw = 0.0;
  //      double yaw_diff = 0.0;
  //        /***tracking mode?***/
  ////        if (target_msg.tracking) {
  //        if (1) {
  //            current_yaw = std::atan2(current_position.y() - odom2pitch.transform.translation.y, current_position.x()
  //            - odom2pitch.transform.translation.x);
  //            /***两角最小距离***/
  //            yaw_diff = angles::shortest_angular_distance(last_yaw_, current_yaw);
  ////            yaw_diff = 0;
  //
  //            if (std::abs(yaw_diff) > max_jump_angle) {
  //                jump_count_++;
  //
  //                if (jump_count_ > 1 && std::signbit(yaw_diff) == std::signbit(last_jump_yaw_diff_)) {
  //                    target_spinning_ = true;
  //                    jump_period_ = time_after_jumping;
  //                }
  //
  //                last_jump_time_ = current_time;
  //                last_jump_position_ = current_position;
  //                last_jump_yaw_diff_ = yaw_diff;
  //            }
  //
  //            last_yaw_ = current_yaw;
  //        }
  //        /***是否开火***/
  //        if (target_spinning_) {
  //            if (time_after_jumping / jump_period_ < allow_following_range) {
  //                suggest_fire_ = true;
  //            } else {
  //                target_msg.target_pos.x = last_jump_position_.x();
  //                target_msg.target_pos.y = last_jump_position_.y();
  //                target_msg.target_pos.z = last_jump_position_.z();
  //                /*** ???***/
  //                target_msg.target_vel.x = 0;
  //                target_msg.target_vel.y = 0;
  //                target_msg.target_vel.z = 0;
  //
  //                suggest_fire_ = false;
  //            }
  //        } else {
  //            /***tracking mode?***/
  ////            suggest_fire_ = target_msg.tracking;
  //            suggest_fire_ = true;
  //        }
  //
  //        /*** ?? ***/
  //        // Update spin_info_msg
  ////        spin_info_msg.header = target_msg.header;
  ////        spin_info_msg.target_spinning = target_spinning_;
  ////        spin_info_msg.suggest_fire = target_msg.suggest_fire;
  ////        spin_info_msg.jump_count = jump_count_;
  ////        spin_info_msg.yaw_diff = yaw_diff;
  ////        spin_info_msg.jump_period = jump_period_;
  ////        spin_info_msg.time_after_jumping = time_after_jumping;
  //
  //        ROS_INFO("target_spinning %d", target_spinning_);
  //        ROS_INFO("suggest_fire %d", suggest_fire_);
  //        ROS_INFO("jump_count %d", jump_count_);
  //        ROS_INFO("current_time %lf", current_time.toSec());
  //        ROS_INFO("last_jump_time %lf", last_jump_time_.toSec());
  //        ROS_INFO("current_yaw %f", current_yaw);
  //        ROS_INFO("last_yaw_ %f", last_yaw_);
  //        ROS_INFO("yaw_diff %f", yaw_diff);
  //        ROS_INFO("jump_period %f", jump_period_);
  //        ROS_INFO("time_after_jumping %f", time_after_jumping);
}

}  // namespace rm_forecast