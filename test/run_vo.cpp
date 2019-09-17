// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/QuaternionStamped.h>




class SensorGrabber
{
public:
// 	SensorGrabber ( DRE_SLAM* vo_slam ) :slam_ ( vo_slam ) {}
	SensorGrabber ( myslam::VisualOdometry* vo_slam ) :slam_ ( vo_slam ) {}
	
	void grabRGBD ( const sensor_msgs::ImageConstPtr& msg_rgb,const sensor_msgs::ImageConstPtr& msg_depth ) 
	{
		// Get images.
		cv_bridge::CvImageConstPtr cv_ptr_rgb = cv_bridge::toCvShare ( msg_rgb );
		cv_bridge::CvImageConstPtr cv_ptr_depth  = cv_bridge::toCvShare ( msg_depth );
		
		// Add RGB-D images.
		slam_->addRGBD ( cv_ptr_rgb->image, cv_ptr_depth->image, cv_ptr_rgb->header.stamp.toSec() );
		
	}// grabRGBD
	
	void grabEncoder ( const geometry_msgs::QuaternionStamped::ConstPtr& en_ptr ) 
	{
		
		// Extract left and right encoder measurements.
		double enl1 = en_ptr->quaternion.x;
		double enl2 = en_ptr->quaternion.y;
		double enr1 = en_ptr->quaternion.z;
		double enr2 = en_ptr->quaternion.w;
		
		// Calculate left and right encoder.
		double enl = 0.5* ( enl1 + enl2 );
		double enr = 0.5* ( enr1 + enr2 );
		double  ts= en_ptr->header.stamp.toSec();
		
		// Check bad data.
		{
			
			if ( last_enl_ == 0 && last_enr_ == 0 ) {
				last_enl_ = enl;
				last_enr_ = enr;
				return;
			}
			
			double delta_enl = fabs ( enl - last_enl_ );
			double delta_enr = fabs ( enr - last_enr_ );
			
			const double delta_th = 4000;
			
			if ( delta_enl > delta_th || delta_enr > delta_th ) {
				std::cout << "\nJUMP\n";
				return;
			}
			
			last_enl_ = enl;
			last_enr_ = enr;
		}
		
		// Add encoder measurements.
		slam_->addEncoder ( enl, enr, ts );
	}// grabEncoder
	
private:
	myslam::VisualOdometry* slam_;
	double last_enl_ = 0;
	double last_enr_ = 0;
};
int main ( int argc, char** argv )
{
    // Init ROS
    ros::init ( argc, argv, "DRE_SLAM" );
    ros::start();
    ros::NodeHandle nh;

// Init SLAM system.

    myslam::VisualOdometry slam();
	//     myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );
	// Sub topics.
    SensorGrabber sensors ( &slam );
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub ( nh, slam.cam_rgb_topic_, 1 );
    message_filters::Subscriber<sensor_msgs::Image> depth_sub ( nh, slam.cam_depth_topic_, 1 );
	
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync ( sync_pol ( 5 ), rgb_sub,depth_sub );
	
    sync.registerCallback ( boost::bind ( &SensorGrabber::grabRGBD,&sensors,_1,_2 ) ); //
	
    ros::Subscriber encoder_sub = nh.subscribe ( slam.encoder_topic_, 1, &SensorGrabber::grabEncoder,&sensors );
	
    std::cout << "\n\nDRE-SLAM Started\n\n";
    ros::spin();
    
    // System Stoped.
    std::cout << "\n\nDRE-SLAM Stoped\n\n";	
    return 0;
}

























































// int main ( int argc, char** argv )
// {
//     if ( argc != 2 )
//     {
//         cout<<"usage: run_vo parameter_file"<<endl;
//         return 1;
//     }
// 
//     myslam::Config::setParameterFile ( argv[1] );
//     myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );
// 
//     string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
//     cout<<"dataset: "<<dataset_dir<<endl;
//     ifstream fin ( dataset_dir+"/associate.txt" );
//     if ( !fin )
//     {
//         cout<<"please generate the associate file called associate.txt!"<<endl;
//         return 1;
//     }
// 
//     vector<string> rgb_files, depth_files;
//     vector<double> rgb_times, depth_times;
//     while ( !fin.eof() )
//     {
//         string rgb_time, rgb_file, depth_time, depth_file;
//         fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
//         rgb_times.push_back ( atof ( rgb_time.c_str() ) );
//         depth_times.push_back ( atof ( depth_time.c_str() ) );
//         rgb_files.push_back ( dataset_dir+"/"+rgb_file );
//         depth_files.push_back ( dataset_dir+"/"+depth_file );
// 
//         if ( fin.good() == false )
//             break;
//     }
// 
//     myslam::Camera::Ptr camera ( new myslam::Camera );
//     
//     // visualization
//     cv::viz::Viz3d vis("Visual Odometry");
//     cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
//     cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
//     cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
//     vis.setViewerPose( cam_pose );
//     
//     world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
//     camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
//     vis.showWidget( "World", world_coor );
//     vis.showWidget( "Camera", camera_coor );
// 
//     cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
//     for ( int i=0; i<rgb_files.size(); i++ )
//     {
//         Mat color = cv::imread ( rgb_files[i] );
//         Mat depth = cv::imread ( depth_files[i], -1 );
//         if ( color.data==nullptr || depth.data==nullptr )
//             break;
//         myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
//         pFrame->camera_ = camera;
//         pFrame->color_ = color;
//         pFrame->depth_ = depth;
//         pFrame->time_stamp_ = rgb_times[i];
// 
//         boost::timer timer;
//         vo->addFrame ( pFrame );
//         cout<<"VO costs time: "<<timer.elapsed()<<endl;
//         
//         if ( vo->state_ == myslam::VisualOdometry::LOST )
//             break;
//         SE3 Tcw = pFrame->T_c_w_.inverse();
//         
//         // show the map and the camera pose 
//         cv::Affine3d M(
//             cv::Affine3d::Mat3( 
//                 Tcw.rotation_matrix()(0,0), Tcw.rotation_matrix()(0,1), Tcw.rotation_matrix()(0,2),
//                 Tcw.rotation_matrix()(1,0), Tcw.rotation_matrix()(1,1), Tcw.rotation_matrix()(1,2),
//                 Tcw.rotation_matrix()(2,0), Tcw.rotation_matrix()(2,1), Tcw.rotation_matrix()(2,2)
//             ), 
//             cv::Affine3d::Vec3(
//                 Tcw.translation()(0,0), Tcw.translation()(1,0), Tcw.translation()(2,0)
//             )
//         );
//         cv::imshow("image", color );
//         cv::waitKey(1);
//         vis.setWidgetPose( "Camera", M);
//         vis.spinOnce(1, false);
//     }
// 
//     return 0;
// }