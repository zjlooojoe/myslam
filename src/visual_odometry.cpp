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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"
#include "encoder.h"
#include "runtimer.h"
#include<mutex>
#include <ceres/ceres.h>
namespace myslam
{
  //----------------------------------------------------
Frame* VisualOdometry::getNewFrame()
{
    std::unique_lock<mutex> lock ( mutex_input_frames_ );
    Frame* frame = input_frames_.front();
    input_frames_.pop();
    return  frame;
}
//这里就完成了一frame对应一vector的enc积分，这之前的搞定了，接下来的就可以做标定了（一个优化问题）

void VisualOdometry::addRGBD(const Mat& rgb, const Mat& depth, const double& timestamp)
{
    // Time cost ~ 1 ms.
    Frame* frame = new Frame ( rgb, depth, timestamp);//, feature_detector_, cam_, cfg_ );
    frame->setEncoders ( encoders_f2f_ );
    encoders_f2f_.clear();

    std::unique_lock<mutex> lock ( mutex_input_frames_ );
    input_frames_.push ( frame );
}

void VisualOdometry::addEncoder(const double& enc_l, const double& enc_r, const double& timestamp)
{
  encoders_f2f_.push_back ( Encoder ( enc_l, enc_r, timestamp ) );
}

void VisualOdometry::RGBDThread()
{
  int i=0;
  while ( true ) {

    //这个是只要接收到一个rgbd就会开始进入它
        if ( checkNewFrame() ) 
	{
            // get current frame.
            curr_ = getNewFrame();
		
	    myslam::RunTimer t;
            t.start();
			
            // process current frame.
            RGBDProcessing();

	    encoder_kf2f_.getTrr();
	    //现在的打算就是视觉给出一个Tcicj，码盘给出一个Toioj，有  min || Tco * Toioj * Toc  -  Tcicj ||,Tco是我们想要的优化的东西，应该先给它个初值。
            t.stop();
//             std::cout << "Frame " << curr_->id_ << " Tracking time: " << t.duration() << "\n";
	    std::cout << "Frame " << i << " Tracking time: " << t.duration() << "\n";
        }// if new frame come in.
        
        i++;

        usleep ( 3000 ); // sleep 3 ms.
    } // while true.
}

void VisualOdometry::RGBDProcessing()
{
//   if ( ! inited_ ) 
//   {
//     initialization();//这里做图片的初始化，ref=cur，所以我这里要将它改为0.3中的初始化！！！！
//     inited_ = true;
// 
//     // Reset Encoder inegration from ref_keyframe to cur_frame.
//     encoder_kf2f_ = EncoderIntegration ( odom_kl_, odom_kr_, odom_b_, odom_K_ );
//     return;
//   } // if not initialization
// 
//   encoder_kf2f_.addEncoders ( curr_->getEncoders() );
//   
//   //应该在这里将encoder_kf2f这个encoder的积分与视觉的delta做优化！！！2019/7/3 10：51
//   curr_->setPose ( ref_kf_->getSE2Pose() * encoder_kf2f_.getTrr() ); // Set init pose
  
  if ( ! inited_ ) 
  {
//     initialization();//这里做图片的初始化，ref=cur，所以我这里要将它改为0.3中的初始化！！！！
    ref_=curr_;
    Eigen::Matrix3d _R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d _t(0,0,0);
    SE3 se3_rt(_R,_t);
    curr_->T_c_w_=ref_->T_c_w_=se3_rt;
    // extract features from first frame and add them into map
    extractKeyPoints();
    computeDescriptors();
    setRef3DPoints();
    
    inited_ = true;

    // Reset Encoder inegration from ref_keyframe to cur_frame.
    encoder_kf2f_ = EncoderIntegration ( odom_kl_, odom_kr_, odom_b_, odom_K_ );
    return;
  } // if not initialization

  //初始化完成，开始正常运行
  //----camera
  //camera temp = T_c_r_estimated_;
  extractKeyPoints();
  computeDescriptors();
  featureMatching();
  poseEstimationPnP();
  if ( checkEstimatedPose() == true ) // a good estimation
  {
      curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
      ref_ = curr_;
      setRef3DPoints();
  }
  //-----encoder
  encoder_kf2f_.addEncoders ( curr_->getEncoders() );
  //encoder temp = encoder.getTrr();
  
  //应该在这里将encoder_kf2f这个encoder的积分与视觉的delta做优化！！！2019/7/3 10：51
  curr_->setPose ( ref_kf_->getSE2Pose() * encoder_kf2f_.getTrr() ); // Set init pose
  
  
  switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
	//------------------
	Eigen::Matrix3d _R=Eigen::Matrix3d::Identity();
	Eigen::Vector3d _t(0,0,0);
	SE3 se3_rt(_R,_t);
	curr_->T_c_w_=ref_->T_c_w_=se3_rt;
	//-----------------------------------------------------
        map_->insertKeyFrame ( frame );
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
	//下面setref3DPoints这步最后把curr的描述子给ref是为了下面使用featureMatching()！！
	//是为了系统初始化后，系统继续进行的 条件
        setRef3DPoints();
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
            setRef3DPoints();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

bool VisualOdometry::checkNewFrame()
{
  std::unique_lock<mutex> lock ( mutex_input_frames_ );
  return ( !input_frames_.empty() );
}

  //----------------------------------------------------

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_( new cv::flann::LshIndexParams(5,10,2) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ("map_point_erase_ratio");
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
    
    
    rgbd_thread_ = new std::thread ( &VisualOdometry::RGBDThread, this );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
	//------------------
	Eigen::Matrix3d _R=Eigen::Matrix3d::Identity();
	Eigen::Vector3d _t(0,0,0);
	SE3 se3_rt(_R,_t);
	curr_->T_c_w_=ref_->T_c_w_=se3_rt;
	//-----------------------------------------------------
        map_->insertKeyFrame ( frame );
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
	//下面setref3DPoints这步最后把curr的描述子给ref是为了下面使用featureMatching()！！
	//是为了系统初始化后，系统继续进行的 条件
        setRef3DPoints();
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
            setRef3DPoints();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    matcher_flann_.match( descriptors_ref_, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    feature_matches_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            feature_matches_.push_back(m);
        }
    }
    cout<<"good matches: "<<feature_matches_.size()<<endl;
    cout<<"match cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::setRef3DPoints()
{
    // select the features with depth measurements 
    pts_3d_ref_.clear();
    descriptors_ref_ = Mat();
    for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    {
        double d = ref_->findDepth(keypoints_curr_[i]);               
        if ( d > 0)
        {
            Vector3d p_cam = ref_->camera_->pixel2camera(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
            );
            pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
            descriptors_ref_.push_back(descriptors_curr_.row(i));
        }
    }
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    for ( cv::DMatch m:feature_matches_ )
    {
        pts3d.push_back( pts_3d_ref_[m.queryIdx] );
        pts2d.push_back( keypoints_curr_[m.trainIdx].pt );
    }
    
    Mat K = ( cv::Mat_<double>(3,3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_r_estimated_ = SE3(
        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
    
    // using bundle adjustment to optimize the pose 
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_r_estimated_.rotation_matrix(), 
        T_c_r_estimated_.translation()
    ) );
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int>(i,0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement( Vector2d(pts2d[index].x, pts2d[index].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge( edge );
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    T_c_r_estimated_ = SE3 (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame ( curr_ );
}

}

