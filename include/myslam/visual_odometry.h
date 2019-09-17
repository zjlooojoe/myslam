/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include "encoder.h"

#include <opencv2/features2d/features2d.hpp>
#include "encoder_integration.h"
#include "runtimer.h"
#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

namespace myslam 
{
  template <typename T>
inline T NormalizeAngle ( const T& angle_radians )
{
    // Use ceres::floor because it is specialized for double and Jet types.
    T two_pi ( 2.0 * M_PI );
    return angle_radians - two_pi * ceres::floor ( ( angle_radians + T ( M_PI ) ) / two_pi );
}
  //-------------------------------------------------optimizer--------------------------
class calibartion
{
public:
  //输入：视觉一个Tcicj，码盘给出一个Toioj，
  //优化变量：Tco                                                                             min || Tco * Toioj * Toc  -  Tcicj ||
  calibartion ( const Eigen::Matrix4d& T_cicj, const Eigen::Matrix4d& T_oioj,  const Eigen::Matrix3d sqrt_info ) :
        T_cicj_ ( T_cicj ), T_oioj_ ( T_oioj ), sqrt_info_ ( sqrt_info ) {}   
//     calibartion ( const double& delta_x, const double& delta_y, const double& delta_theta, const Eigen::Matrix3d sqrt_info ) :
//         delta_x_ ( delta_x ), delta_y_ ( delta_y ), delta_theta_ ( delta_theta ), sqrt_info_ ( sqrt_info ) {}

    template <typename T>
    bool operator() (T const* const* parameters , T* residuals ) const /*Tco要用double数组传进来*/
    {
      //|| Tco * Toioj * Toc  -  Tcicj ||
      //先像vins那样传入四元数，再在这里转换成矩阵，跟T_cicj_ T_oioj_计算然后再转回四元数
      //double Tco[16]
      /* 
    Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> T_estimated(parameters[0]);
    Eigen::Map<Eigen::VectorXd> resVec(residuals);

    //第一种：用旋转矩阵去做残差 看看DynamicAutoDiffFunction行不行，主要是为了传入矩阵         此时残差6维，rotation=3 translation=3           
    //       这种可以尝试不用协方差，可以美其名曰 暂时把它看作是等权的，觉得只是利用系统起始几帧，没必要考虑加权  要考虑的话，此时的协方差阵要是6维的！！！！！！
    
    //第二种：可以像dre那样，用se2做残差，dx dy d_th(用se2的李代数去表示旋转，se2的李代数应该就是角度theta) 用的costfuntion          此时残差3维 dx dy dth,不考虑dz，
    //       这里的话就还是想他们那样考虑一下协方差吧。先记录直接的残差，然后再乘对应的协方差就可以了
    const Eigen::Matrix3d & dR_c = T_cicj_.block<3,3>(0,0);
    const Eigen::Matrix3d & dR_o = T_oioj_.block<3,3>(0,0);
    const Eigen::Matrix3d & R_co = T_estimated.block<3,3>(0,0);
    const Eigen::Vector3d & dt_c = T_cicj_.block<3,1>(0,3);
    const Eigen::Vector3d & dt_o = T_oioj_.block<3,1>(0,3);
    const Eigen::Vector3d & t_co = T_estimated.block<3,1>(0,3);
    const Eigen::Matrix3d & dR_x = R_co* dR_o* R_co.transpose();
    
    Eigen::Vector3d rotion_err = cvte::Log(dR_c.transpose()* dR_x);//可以将旋转矩阵相乘之后输出为旋转向量 3维的
    Eigen::Vector3d translation_err = dR_c.transpose()*( -dR_x*t_co +  R_co*dt_o + t_co - dt_c);
    resVec<<rotion_err,translation_err;
    resVec<<rotion_err;
    //有协方差的话就
    resVec = sqrt_info_ * resVec;//这样的话sqrt_info_要是六维的   6×6 6×1              
    
    */
    
     //第二种：
     //   Copy data 
    Sophus::SE2 f_Twr2 = frame->getSE2Pose();
    double frame_pose[3] = {f_Twr2.translation() ( 0 ), f_Twr2.translation() ( 1 ), f_Twr2.so2().log() };
    Sophus::SE2 ref_kf_Twr = ref_kf->getSE2Pose();
    double  ref_kf_pose[3] = {ref_kf_Twr.translation() ( 0 ), ref_kf_Twr.translation() ( 1 ), ref_kf_Twr.so2().log() };
    
    
      calibartion::Create ( encoder_kf2f_.getTrr().translation() ( 0 ), encoder_kf2f_.getTrr().translation() ( 1 ), encoder_kf2f_.getTrr().so2().log(), o_sqrt_info );
    problem.AddResidualBlock ( cost_function,
                               encoder_loss,
                               ref_kf_pose, ref_kf_pose + 1, ref_kf_pose +2,
                               frame_pose, frame_pose + 1,  frame_pose +2
                             );

       // ref pose
        T xr = ref_x[0];
        T yr = ref_y[0];
        T thr = ref_th[0];

        // cur pose
        T xc = cur_x[0];
        T yc = cur_y[0];
        T thc = cur_th[0];

        T ob_dx = T ( delta_x_ );
        T ob_dy = T ( delta_y_ );
        T ob_dth = T ( delta_theta_ );

        T tmp_dx = xc - xr;
        T tmp_dy = yc - yr;

	//this
        T dx = cos ( thr ) *tmp_dx + sin ( thr ) *tmp_dy;
        T dy = -sin ( thr ) *tmp_dx + cos ( thr ) *tmp_dy;

        T ex = dx - ob_dx;
        T ey = dy - ob_dy;
        T eth = NormalizeAngle ( thc - thr - ob_dth );// [(cur_th - ref_th) - delta_enc_th]

        residuals[0] = T ( sqrt_info_ ( 0,0 ) ) * ex + T ( sqrt_info_ ( 0,1 ) ) * ey + T ( sqrt_info_ ( 0,2 ) ) * eth ;
        residuals[1] = T ( sqrt_info_ ( 1,0 ) ) * ex + T ( sqrt_info_ ( 1,1 ) ) * ey + T ( sqrt_info_ ( 1,2 ) ) * eth ;
        residuals[2] = T ( sqrt_info_ ( 2,0 ) ) * ex + T ( sqrt_info_ ( 2,1 ) ) * ey + T ( sqrt_info_ ( 2,2 ) ) * eth ;
     
     
      return true;
    }
    
    static ceres::CostFunction* Create ( const Eigen::Matrix4d& T_cicj, const Eigen::Matrix4d& T_oioj,  const Eigen::Matrix3d sqrt_info ) {
        return ( new ceres::DynamicAutoDiffCostFunction<calibartion, 3, 1, 1, 1, 1, 1, 1>
                 ( new calibartion ( T_cicj, T_oioj, sqrt_info ) ) );
    
//     bool operator() ( const T* const ref_x, const T* const ref_y, const T* const ref_th,
//                       const T* const cur_x, const T* const cur_y, const T* const cur_th,
//                       T* residuals ) const {
// 
//         // ref pose
//         T xr = ref_x[0];
//         T yr = ref_y[0];
//         T thr = ref_th[0];
// 
//         // cur pose
//         T xc = cur_x[0];
//         T yc = cur_y[0];
//         T thc = cur_th[0];
// 
//         T ob_dx = T ( delta_x_ );
//         T ob_dy = T ( delta_y_ );
//         T ob_dth = T ( delta_theta_ );
// 
//         T tmp_dx = xc - xr;
//         T tmp_dy = yc - yr;
// 
// 	//this
//         T dx = cos ( thr ) *tmp_dx + sin ( thr ) *tmp_dy;
//         T dy = -sin ( thr ) *tmp_dx + cos ( thr ) *tmp_dy;
// 
//         T ex = dx - ob_dx;
//         T ey = dy - ob_dy;
//         T eth = NormalizeAngle ( thc - thr - ob_dth );// [(cur_th - ref_th) - delta_enc_th]
// 
//         residuals[0] = T ( sqrt_info_ ( 0,0 ) ) * ex + T ( sqrt_info_ ( 0,1 ) ) * ey + T ( sqrt_info_ ( 0,2 ) ) * eth ;
//         residuals[1] = T ( sqrt_info_ ( 1,0 ) ) * ex + T ( sqrt_info_ ( 1,1 ) ) * ey + T ( sqrt_info_ ( 1,2 ) ) * eth ;
//         residuals[2] = T ( sqrt_info_ ( 2,0 ) ) * ex + T ( sqrt_info_ ( 2,1 ) ) * ey + T ( sqrt_info_ ( 2,2 ) ) * eth ;
// 
//         return true;
//     }
//     static ceres::CostFunction* Create ( const double& delta_x, const double& delta_y, const double& delta_theta, const Eigen::Matrix3d sqrt_info ) {
//         return ( new ceres::AutoDiffCostFunction<calibartion, 3, 1, 1, 1, 1, 1, 1>
//                  ( new calibartion ( delta_x, delta_y, delta_theta, sqrt_info ) ) );
    }

private:
    const double delta_x_, delta_y_, delta_theta_;
    const Eigen::Matrix3d sqrt_info_;
    const Eigen::Matrix4d& T_cicj_, T_oioj_;
}; //class

//----------------------------------------------------------
  
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    
    Frame::Ptr  ref_;       // reference key-frame 
    Frame::Ptr  curr_;      // current frame 
    
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    vector<cv::Point3f>     pts_3d_ref_;        // 3d points in reference frame 
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame 
    Mat                     descriptors_ref_;   // descriptor in reference frame 
    vector<cv::DMatch>      feature_matches_;   // feature matches 
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
   
    SE3 T_c_r_estimated_;    // the estimated pose of current frame 
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    
    // parameters 
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double  map_point_erase_ratio_; // remove map point ratio
    
    void addRGBD(const cv::Mat& rgb, const cv::Mat& depth, const double& timestamp);
    void addEncoder(const double& enc_l, const double& enc_r, const double& timestamp);
    Frame* getNewFrame();
    void RGBDThread();
    void RGBDProcessing(); 
    bool checkNewFrame();
    std::thread* rgbd_thread_;
    
    // raw encoder data.
    std::vector<Encoder> encoders_f2f_;

    // integrated encoder between the reference KF and the current frame.
    EncoderIntegration encoder_kf2f_;
    std::mutex mutex_input_frames_;
    std::queue<Frame*>input_frames_;
    bool inited_;
    
    
/**** Camera ****/
string cam_rgb_topic_ ="/kinect2/qhd/image_color";
string cam_depth_topic_="/kinect2/qhd/image_depth_rect";
double cam_fx_=525.2866213437447  ;
double cam_fy_=525.2178123117577  ;
double cam_cx_=472.85738972861157  ;
double cam_cy_=264.77181506420266  ;
double cam_k1_=0.04160142651680036  ;
double cam_k2_=-0.04771035303381654  ;
double cam_p1_=-0.0032638387781624705  ;
double cam_p2_=-0.003985120051161831  ;
double cam_k3_=0.01110263483766991  ;
int cam_height_=540  ;
int cam_width_=960    ;	
double cam_depth_factor_; 	// Depth scale factor.
double cam_dmax_;			// Max depth value to be used.
double cam_dmin_;			// Min depth value to be used.
double cam_fps_;			// Camera FPs.

/**** Robot intrinsic and extrinsic ****/
std::string encoder_topic_="/rbot/encoder";
	
double odom_kl_=4.0652e-5; 	// left wheel factor
double odom_kr_=4.0668e-5; 	// right wheel factor
double odom_b_=0.3166 ; 	// wheel space
double odom_K_=0.008 ; 	// Noise factor.
Sophus::SE3 Trc_; 	// Extrinsic parameter. Translation from the camera to the robot.
    
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 
    
protected:  
    // inner operation 
    void extractKeyPoints();
    void computeDescriptors(); 
    void featureMatching();
    void setRef3DPoints();
    void poseEstimationPnP(); 
    
    void addKeyFrame();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    
};
}

#endif // VISUALODOMETRY_H
