//
// Created by ljt666666 on 22-10-9.
//
#include "../include/forecast_node.h"
#include "../include/tracker.h"

namespace rm_forecast
{
    Tracker::Tracker(const KalmanFilterMatrices & kf_matrices)
            : tracker_state(LOST),
              tracking_id(0),
              kf_matrices_(kf_matrices),
              tracking_velocity_(Eigen::Vector3d::Zero())
    {
    }

    void Tracker::init(const float &angle)
    {
        // KF init
        kf_ = std::make_unique<KalmanFilter>(kf_matrices_);
        Eigen::VectorXd init_state(6);
//        const auto position = chosen_armor.pose.position;
//        init_state << position.x, position.y, position.z, 0, 0, 0;
        init_state << angle, 0, 0, 0, 0, 0;
        kf_->init(init_state);

        tracker_state = DETECTING; /***设置装甲板状态为目标识别中，需要更多帧识别到才开始跟踪***/
    }

    void Tracker::update(const float &angle, const double & dt, const double &max_match_distance, const int &tracking_threshold, const int &lost_threshold)
    {
        // KF predict
        kf_matrices_.F(0, 3) = kf_matrices_.F(1, 4) = kf_matrices_.F(2, 5) = dt; /***更新f矩阵***/
        Eigen::VectorXd kf_prediction = kf_->predict(kf_matrices_.F); /***得出本时刻预测值***/

        bool matched = false;
        // Use KF prediction as default target state if no matched armor is found
        target_state = kf_prediction; /***本时刻预测值***/


        Eigen::Vector3d position_vec(
                angle, 0, 0);
        /***只有这个时候才做卡尔曼滤波的更新部分***/
        target_state = kf_->update(position_vec);
            /***如果离预测目标最近的装甲板的距离大于阈值，则寻找另一个id相同的装甲板重新作为追踪目标，并把上一个追踪目标的速度给它***/
    }

}  // namespace rm_auto_aim