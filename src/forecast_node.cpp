//
// Created by ljt666666 on 22-10-9.
//

#include "../include/forecast_node.h"
#include <pluginlib/class_list_macros.h>
#include <ros/callback_queue.h>

using namespace cv;
using namespace std;

PLUGINLIB_EXPORT_CLASS(rm_forecast::Forecast_Node, nodelet::Nodelet)

namespace rm_forecast {
void Forecast_Node::onInit() {
  ros::NodeHandle &nh = getMTPrivateNodeHandle();
  static ros::CallbackQueue my_queue;
  nh.setCallbackQueue(&my_queue);
  initialize(nh);
  my_thread_ = std::thread([]() {
    ros::SingleThreadedSpinner spinner;
    spinner.spin(&my_queue);
  });
}

void Forecast_Node::initialize(ros::NodeHandle &nh) {
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
  q.diagonal() << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;

  // R - measurement noise covariance matrix
  Eigen::DiagonalMatrix<double, 3> r;
  r.diagonal() << 0.05, 0.05, 0.05;

  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, 6> p;
  p.setIdentity();

  kf_matrices_ =
      KalmanFilterMatrices{f, h, q, r, p}; /***初始化卡尔曼滤波初始参数***/

  if(!nh.getParam("max_match_distance", max_match_distance_))
      ROS_WARN("No max match distance specified");
    ROS_INFO("66%lf", max_match_distance_);
  if(!nh.getParam("tracking_threshold", tracking_threshold_))
        ROS_WARN("No tracking threshold specified");
  if(!nh.getParam("lost_threshold", lost_threshold_))
        ROS_WARN("No lost threshold specified");

    if(!nh.getParam("max_jump_angle", max_jump_angle_))
        ROS_WARN("No max_jump_angle specified");
    if(!nh.getParam("max_jump_period", max_jump_period_))
        ROS_WARN("No max_jump_period_ specified");
    if(!nh.getParam("allow_following_range", allow_following_range_))
        ROS_WARN("No allow_following_range specified");

    if(!nh.getParam("y_thred", y_thred_))
        ROS_WARN("No y_thred specified");
    if(!nh.getParam("fly_time", fly_time_))
        ROS_WARN("No fly_time specified");
    if(!nh.getParam("allow_following_range", allow_following_range_))
        ROS_WARN("No allow_following_range specified");

  tracker_ = std::make_unique<Tracker>(kf_matrices_);

  spin_observer_ = std::make_unique<SpinObserver>();

  forecast_cfg_srv_ = new dynamic_reconfigure::Server<rm_forecast::ForecastConfig>(ros::NodeHandle(nh_, "rm_forecast"));
  forecast_cfg_cb_ = boost::bind(&Forecast_Node::forecastconfigCB, this, _1, _2);
  forecast_cfg_srv_->setCallback(forecast_cfg_cb_);

  tf_buffer_ = new tf2_ros::Buffer(ros::Duration(10));
  tf_listener_ = new tf2_ros::TransformListener(*tf_buffer_);
//  targets_sub_ =
//      nh.subscribe("/detection", 1, &Forecast_Node::speedCallback, this);
//    targets_sub_ =
//            nh.subscribe("/detection", 1, &Forecast_Node::outpostCallback, this);
    targets_sub_ =
            nh.subscribe("/processor/result_msg", 1, &Forecast_Node::pointsCallback, this);
  track_pub_ = nh.advertise<rm_msgs::TrackData>("/track", 10);

    std::vector<float> intrinsic;
    std::vector<float> distortion;
    if(!nh.getParam("/forecast/camera_matrix/data", intrinsic))
        ROS_WARN("No cam_intrinsic_mat_k specified");
    if(!nh.getParam("/forecast/distortion_coefficients/data", distortion))
        ROS_WARN("No distortion specified");
    if(!nh.getParam("is_reproject", is_reproject_))
        ROS_WARN("No is_reproject specified");
    if(!nh.getParam("re_fly_time", re_fly_time_))
        ROS_WARN("No re_fly_time specified");

    Eigen::MatrixXd mat_intrinsic(3, 3);
    initMatrix(mat_intrinsic,intrinsic);
    eigen2cv(mat_intrinsic,m_intrinsic_);

    cam_intrinsic_mat_k_ = cv::Matx<float, 3, 3>(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4], intrinsic[5],
                                                 intrinsic[6], intrinsic[7], intrinsic[8]);
    std::cout << "intrinsic maxtric is: " << cam_intrinsic_mat_k_ << std::endl;
    dist_coefficients_ = cv::Matx<float, 1, 5>(distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]);

    draw_sub_ = nh.subscribe("/hk_camera/camera/image_raw", 1, &Forecast_Node::drawCallback, this);
    draw_pub_ = it_->advertise("reproject_image", 1);
}

void Forecast_Node::forecastconfigCB(rm_forecast::ForecastConfig &config,
                                     uint32_t level) {
//          target_type_ = config.target_color;
    ///track
  max_match_distance_ = config.max_match_distance;
  tracking_threshold_ = config.tracking_threshold;
  lost_threshold_ = config.lost_threshold;

  ///spin_observer
    max_jump_angle_ = config.max_jump_angle;
    max_jump_period_ = config.max_jump_period;
    allow_following_range_ = config.allow_following_range;

  ///outpost
    forecast_readied_ = config.forecast_readied;
    reset_ = config.reset;
    min_target_quantity_ = config.min_target_quantity;
    line_speed_ = config.line_speed;
    z_c_ = config.z_c;
    outpost_radius_ = config.outpost_radius;
    rotate_speed_ = config.rotate_speed;
    y_thred_ = config.y_thred;
    fly_time_ = config.fly_time;

    ///reproject
    is_reproject_ = config.is_reproject;
    re_fly_time_ = config.re_fly_time;

    ///windmill_kalman
    delay_time_ = config.delay_time;

}

void Forecast_Node::pointsCallback(const rm_msgs::TargetDetectionArray::Ptr &msg){
    rm_msgs::TrackData track_data;
    track_data.header.frame_id = "odom";
    track_data.header.stamp = msg->header.stamp;
    track_data.id = 0;

    if (msg->detections.empty()){
        track_pub_.publish(track_data);
        return;
    }

//        ROS_INFO("POINTS");
    int32_t data[4 * 2];                        // data of 4 2D points
    for (const auto &detection : msg->detections) {
        memcpy(&data[0], &detection.pose.orientation.x, sizeof(int32_t) * 2);
        memcpy(&data[2], &detection.pose.orientation.y, sizeof(int32_t) * 2);
        memcpy(&data[4], &detection.pose.orientation.z, sizeof(int32_t) * 2);
        memcpy(&data[6], &detection.pose.orientation.w, sizeof(int32_t) * 2);
        memcpy(&data[8], &detection.pose.position.x, sizeof(int32_t) * 2);
    }

    std::vector<Point2f> pic_points;
    for (int i = 0; i < 4; i++){
        Point2f point;
        point.x = float(data[2*i]);
        point.y = float(data[2*i+1]);
        pic_points.emplace_back(point);
            ROS_INFO("x%f, y%f", point.x, point.y);
    }
    Point2f r_2d;
    r_2d.x = data[8];
    r_2d.y = data[9];

    Target hit_target = pnp(pic_points);
    Point2f target2d = reproject(hit_target.tvec);

    InfoTarget info_target;
    float angle;
    if(tracking_){
    Point2d vec1(last_target2d_.x - last_r2d_.x, last_target2d_.y - last_r2d_.y);
    Point2d vec2(target2d.x - r_2d.x, target2d.y - r_2d.y);
//        Point2d vec1(last_target2d_.x , last_target2d_.y);
//        Point2d vec2(target2d.x, target2d.y);
        auto costheta = static_cast<float>(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2)));
        angle = acos(costheta);
        ROS_INFO("angle:%f", angle);
        info_target.stamp = msg->header.stamp;
        info_target.angle = angle;
    }

    last_target2d_ = r_2d;
    last_target2d_ = target2d;

    if (tracker_->tracker_state == Tracker::LOST) {
        tracker_->init(0.1);
        //            target_msg.tracking = false;
        tracking_ = false;
    }
        /***是其他状态则更新tracker***/
    else {
        // Set dt
        dt_ = (msg->header.stamp - last_time_).toSec();
        // Update state
        tracker_->update(info_target.angle, dt_, max_match_distance_, tracking_threshold_, lost_threshold_);
        tracking_ = true;
    }
    last_time_ = msg->header.stamp;

////    track_data.target_pos.x = tracker_->target_state(0);
////    track_data.target_pos.y = tracker_->target_state(1);
////    track_data.target_pos.z = tracker_->target_state(2);
////    track_data.target_vel.x = tracker_->target_state(3);
////    track_data.target_vel.y = tracker_->target_state(4);
////    track_data.target_vel.z = tracker_->target_state(5);

    if(!tracking_)
        return;
    double params[4];
    params[3] = abs(tracker_->target_state(3));
    double t0 = 0;
    double t1 = delay_time_;
    int mode = 0;
    std::vector<double> hit_point = calcAimingAngleOffset(hit_target, params, t0, t1, mode);

    ROS_INFO("x%f, y%f, z%f", hit_point[0], hit_point[1], hit_point[2]);

    rm_msgs::TargetDetection detection_temp;
    detection_temp.pose.position.x = hit_point[0];
    detection_temp.pose.position.y = hit_point[1];
    detection_temp.pose.position.z = hit_point[2];

    geometry_msgs::PoseStamped pose_in;
    geometry_msgs::PoseStamped pose_out;
    pose_in.header.frame_id = "camera_optical_frame";
    pose_in.header.stamp = msg->header.stamp;
    pose_in.pose = detection_temp.pose;

    try
    {
        geometry_msgs::TransformStamped transform = tf_buffer_->lookupTransform(
                "odom", pose_in.header.frame_id, msg->header.stamp, ros::Duration(1));

        tf2::doTransform(pose_in.pose, pose_out.pose, transform);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
    }
//    ROS_INFO_STREAM(pose_out.pose.position.x
//                    << ",y:" << pose_out.pose.position.y
//                    << ",z:" << pose_out.pose.position.z);
    detection_temp.pose = pose_out.pose;

    track_data.id = 3;
    track_data.target_pos.x = detection_temp.pose.position.x;
    track_data.target_pos.y = detection_temp.pose.position.y;
    track_data.target_pos.z = detection_temp.pose.position.z;
    track_data.target_vel.x = angle;
    track_data.target_vel.y = tracker_->target_state(3);
    track_data.target_vel.z = 0;

    track_pub_.publish(track_data);
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
        points_world = {
                {-0.066,-0.027,0},
                {-0.066,0.027,0},
                {0.066,0.027,0},
                {0.066,-0.027,0}};
//        points_world = {
//                {-0.14,-0.08,0},
//                {-0.14,0.08,0},
////         {0,-0.565,-0.05},
//                {0.14,0.08,0},
//                {0.14,-0.08,0}};

        Mat rvec;
        Mat rmat;
        Mat tvec;
        Eigen::Matrix3d rmat_eigen;
        Eigen::Vector3d R_center_world = {0,-0.7,-0.05};
        Eigen::Vector3d tvec_eigen;
        Eigen::Vector3d coord_camera;

        solvePnP(points_world, points_pic, cam_intrinsic_mat_k_, dist_coefficients_, rvec, tvec, false, SOLVEPNP_ITERATIVE);

        std::array<double, 3> trans_vec = tvec.reshape(1, 1);
        ROS_INFO("x:%f, y:%f, z:%f", trans_vec[0], trans_vec[1], trans_vec[2]);

        Target result;
//        //Pc = R * Pw + T
        cv::Rodrigues(rvec, rmat); /***罗德里格斯变换，把旋转向量转换为旋转矩阵***/
        cv::cv2eigen(rmat, rmat_eigen);/***cv转成eigen格式***/
        cv::cv2eigen(tvec, tvec_eigen);
//
        result.rmat = rmat_eigen;
        result.tvec = tvec_eigen;

        return result;
    }

    std::vector<double> Forecast_Node::calcAimingAngleOffset(Target& object, double params[4], double t0, double t1 , int mode)
    {
        auto a = params[0];
        auto omega = params[1];
        auto theta = params[2];
        auto b = params[3];
        double theta1;
        double theta0;
         cout<<"t1: "<<t1<<endl;
         cout<<"t0: "<<t0<<endl;
        //f(x) = a * sin(ω * t + θ) + b
        //对目标函数进行积分
        if (mode == 0)//适用于小符模式
        {
            theta0 = b * t0;
            theta1 = b * t1;
        }
        else
        {
            theta0 = (b * t0 - (a / omega) * cos(omega * t0 + theta));
            theta1 = (b * t1 - (a / omega) * cos(omega * t1 + theta));
        }
         cout<<(theta1 - theta0) * 180 / CV_PI<<endl;
//        return theta1 - theta0;
        double theta_offset = theta1 - theta0;
        ROS_INFO("theta1%f, theta0%f, b%f", theta1, theta0, b);
        Eigen::Vector3d hit_point_world = {-sin(theta_offset) * fan_length_, -(cos(theta_offset) - 1) * fan_length_,0};
        Eigen::Vector3d hit_point_cam= (object.rmat * hit_point_world) + object.tvec;
        std::cout << "hit_point_world.transpose() = \n" << hit_point_world.transpose() << std::endl;
        std::cout << "hit_point_cam.transpose() = \n" << hit_point_cam.transpose() << std::endl;
        std::vector<double> hit_points;
        hit_points.emplace_back(hit_point_cam.transpose()[0]);
        hit_points.emplace_back(hit_point_cam.transpose()[1]);
        hit_points.emplace_back(hit_point_cam.transpose()[2]);

        target2d_ = reproject(hit_point_cam);

        return hit_points;
    }

    /**
    * @brief 重投影
    *
    * @param xyz 目标三维坐标
    * @return cv::Point2f 图像坐标系上坐标(x,y)
    */
    cv::Point2f Forecast_Node::reproject(Eigen::Vector3d &xyz)
    {
        Eigen::Matrix3d mat_intrinsic;
        cv2eigen(m_intrinsic_, mat_intrinsic);
        //(u,v,1)^T = (1/Z) * K * (X,Y,Z)^T
        auto result = (1.f / xyz[2]) * mat_intrinsic * (xyz);//解算前进行单位转换
        return cv::Point2f(result[0], result[1]);
    }

    void Forecast_Node::drawCallback(const sensor_msgs::ImageConstPtr& img){
        if(!is_reproject_)
            return;

        cv::Mat origin_img = cv_bridge::toCvShare(img, "bgr8")->image;
        circle(origin_img, target2d_, 10, cv::Scalar(0, 0, 255), -1, 2);
        draw_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", origin_img).toImageMsg());
    }

    template<typename T>
    bool Forecast_Node::initMatrix(Eigen::MatrixXd &matrix,std::vector<T> &vector)
    {
        int cnt = 0;
        for(int row = 0;row < matrix.rows();row++)
        {
            for(int col = 0;col < matrix.cols();col++)
            {
                matrix(row,col) = vector[cnt];
                cnt++;
            }
        }
        return true;
    }

} // namespace rm_forecast
