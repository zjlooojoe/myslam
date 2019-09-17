#include <iostream>
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 
#include <ceres/ceres.h>
#include <chrono>
#include <opencv2/features2d/features2d.hpp>// extract ORB
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.h>
#include <sophus/se2.h>
#include <sophus/so3.h>
#include<opencv2/core/eigen.hpp>
//using Sophus::SE3;
//using Sophus::SO3;
/*问题摘录
 * 之前运行的很慢是因为没有做
 * last_keypoint.clear();
   curr_keypoint.clear();
   good_feature_matches.clear();
 * last_keypoint_in_cameraframe.clear();
 * curr_keypoint_in_cameraframe.clear(); 
 这些中间变量的清空，导致他们的大小持续变大，把之前的信息都摘录了下来
 * 
 * 
 *  
 */
using namespace std;
class Frame
{
public:
  Frame(){}
  Frame(cv::Mat image)
  {
	cv::Ptr<cv::ORB> orb_;
	//orb_=cv::ORB::create();
	//slambook
	orb_=cv::ORB::create ( 500, 1.2f, 8 );
	orb_->detect(image,keypoint_);
	orb_->compute(image,keypoint_,descriptors);
  }
  ~Frame(){}
  
  vector<cv::KeyPoint> keypoint_;
  cv::Mat descriptors;
  //Sophus::SE3 Tcw;
  Eigen::Matrix4d mTcw;
  cv::Mat mTcw_F;
  cv::Mat R,t;
  vector<cv::Point3d> _3dpoint;
  
};
cv::Mat im;
Frame frame,currframe,lastframe,initframe;
vector<cv::DMatch> good_feature_matches;
vector<cv::Point2f> last_keypoint;
vector<cv::Point2f> curr_keypoint;
vector<cv::Point2f> last_keypoint_in_cameraframe;
vector<cv::Point2f> curr_keypoint_in_cameraframe;
cv::Mat Rcr,tcr;//using in poseEstimate
//全局变量，用来将Rcr tcr 组建起来，作为当前 poseEstimate 恢复的r->c的 T
Eigen::Matrix4d mTcr=Eigen::Matrix4d::Identity();

//Sophus::SE3 Tcr_;
cv::Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
double focal_length = 521;			//相机焦距, TUM dataset标定值


cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

bool initstal=false;

cv::Point2f pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}
void triangulation(const vector< cv::KeyPoint >& keypoint_1, const vector< cv::KeyPoint >& keypoint_2, const std::vector< cv::DMatch >& matches,
		   const cv::Mat& Rcr, const cv::Mat& tcr, vector< cv::Point3d >& _3dpoints )
{
  //这个T1也可能不是每次都是单位的，可能含义是lastframe的TCW，不过现在先这样设置，出错的话再改
       
      cv::Mat T1=(cv::Mat_<float>(3,4)<<  
	1,0,0,0,
	0,1,0,0,
	0,0,1,0		
      );
      cv::Mat T2=(cv::Mat_<float>(3,4)<<
      
	Rcr.at<double>(0,0),Rcr.at<double>(0,1),Rcr.at<double>(0,2),tcr.at<double>(0,0),
	Rcr.at<double>(1,0),Rcr.at<double>(1,1),Rcr.at<double>(1,2),tcr.at<double>(1,0),
	Rcr.at<double>(2,0),Rcr.at<double>(2,1),Rcr.at<double>(2,2),tcr.at<double>(2,0)	
      );
      //vector<cv::Point2f> pts_1, pts_2;
      last_keypoint_in_cameraframe.clear();
      curr_keypoint_in_cameraframe.clear();
      for ( cv::DMatch m:matches )
      {
	  // 将像素坐标转换至相机坐标
	  last_keypoint_in_cameraframe.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
	  curr_keypoint_in_cameraframe.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
      }
      //还要恢复出3d点！～深度
      //第三第四个参数需要是归一化坐标
      //cv::triangulatePoints(T1,T2,last_keypoint,curr_keypoint,points);
      cv::Mat pts_4d;
      cv::triangulatePoints(T1,T2,last_keypoint_in_cameraframe,curr_keypoint_in_cameraframe,pts_4d);
      for(int i=0;i<pts_4d.cols;i++)
      {
	cv::Mat x = pts_4d.col(i);
	x /= x.at<float>(3,0); // 归一化
	cv::Point3d p (
	x.at<float>(0,0), 
	x.at<float>(1,0), 
	x.at<float>(2,0) 
	);
	_3dpoints.push_back( p );
      }
}
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
void featurematch()
{
  //TODO	      
	      /*
	       * 这里还没做特征间的匹配，可以直接用暴力匹配，但是会有很多误匹配，然后可以添加下面这些操作提高特征匹配精度
	       * 具体匹配思路，根据选取的第一帧的特征点的位置作为输入，然后再第二帧该位置半径r的范围内，寻找可能匹配的点，注意这边的匹配只考虑了原始图像即尺度图像的第一层的特征
	       * 找到了可能的匹配点，下一步进行匹配计算，根据可能匹配特征点的描述子计算距离，确定最佳匹配，
	       * 另外如果考虑特征点的方向，则将第一帧中的特征的方向角度减去对应第二帧的特征的方向角度，将值划分为直方图，则会在0度和360度左右对应的组距比较大，这样就可以对其它相差太大的角度可以进行剔除，
	      */
    vector<cv::DMatch>matches;
     cv::FlannBasedMatcher   matcher_flann_(new cv::flann::LshIndexParams ( 5,10,2 ) );
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    //matcher.match(lastframe.descriptors,currframe.descriptors,matches);
    matcher_flann_.match(lastframe.descriptors,currframe.descriptors,matches);
  //挑选好的匹配点 筛选
    double min_dist=1000,max_dist=0;
  //得到上面各匹配对的描述子距离的最小最大值
    for(int i=0;i<currframe.descriptors.rows;i++)
    {
      double distance=matches[i].distance;
      if(distance<min_dist)min_dist=distance;
      if(distance>=max_dist)max_dist=distance;
    }
     good_feature_matches.clear();
    for(auto m:matches/*int i=0;i<currframe.descriptors.rows;i++*/)
    {
      //if(m.distance<1.5*min_dist)
      if(m.distance<= max ( 1.5*min_dist, 30.0 ))
      {
	good_feature_matches.push_back(m);
      }
    }
  //这里可以多加一些判断，就是看看匹配数有多少，设定一个阈值
    //cout<<good_feature_matches.size()<<endl;
    last_keypoint.clear();
    curr_keypoint.clear();
    for(cv::DMatch m:good_feature_matches)
    {
      last_keypoint.push_back(lastframe.keypoint_[m.queryIdx].pt);
      curr_keypoint.push_back(currframe.keypoint_[m.trainIdx].pt);
    }
    
}
void poseEstimate2d2d()
{
  //要知道这个recoverPose得到的R t是c->r or r->c                                         !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
//     last_keypoint.clear();
//     curr_keypoint.clear();
//   for(cv::DMatch m:good_feature_matches)
//     {
//       last_keypoint.push_back(lastframe.keypoint_[m.queryIdx].pt);
//       curr_keypoint.push_back(currframe.keypoint_[m.trainIdx].pt);
//     }
 
  //cv::Mat E=cv::findEssentialMat(last_keypoint,curr_keypoint,focal_length,principal_point,cv::RANSAC);
  cv::Mat E=cv::findEssentialMat(last_keypoint,curr_keypoint,focal_length,principal_point);
  cv::recoverPose(E,last_keypoint,curr_keypoint,Rcr,tcr,focal_length,principal_point);
 
  cv::cv2eigen<double>(Rcr,R);
  cv::cv2eigen<double>(tcr,t);
//   Sophus::SE3 temp(R,t);
  //Tcr_=temp;
  
//   Eigen::Matrix3d RRR=temp.rotation_matrix(); 
//   Eigen::Vector3d ttt=temp.translation();
  mTcr.block<3,3>(0,0)=R;
  mTcr.block<3,1>(0,3)=t;
  //cout<<mTcr<<endl;
}

/*
 
 vector<cv::Point3d> _3dpoint
 vector<cv::Point2d> _2dpoint
 cv::Mat R
 cv::Mat t
 
 */
void poseEstimate3d2d(vector<cv::Point3d>& _3dpoint,vector<cv::Point2f>& _2dpoint)
{
   cv::Mat rvec, tvec, inliers;
   cv::solvePnPRansac( _3dpoint, _2dpoint, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
//     num_inliers_ = inliers.rows;
//     cout<<"pnp inliers: "<<num_inliers_<<endl;
   Sophus::SE3 Tcr;
   Tcr = Sophus::SE3(
        Sophus::SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Eigen::Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
   mTcr.block<3,3>(0,0)=Tcr.rotation_matrix();
   mTcr.block<3,1>(0,3)=Tcr.translation();
    //Tcr转换到Eigen：：Matrix
    //然后再与上一帧位姿相乘得到当前帧的位姿，看看这样效果好不好
   
}
enum state
  {
    INITSTAL=-1,
    OK=0,
    LOST=1
  };
  
int main(int argc,char** argv)
{
  state mstate=INITSTAL;
  if(argc!=2)
    cout<<"usage run_vo2 config/.yaml"<<endl;
  //读取配置文件，文件中有 图像内参 和 图像路径 
  //读取完之后就开始对图像做处理
    
  cv::FileStorage file_;
  file_=cv::FileStorage(argv[1], cv::FileStorage::READ);
  string dataset_dir=file_["dataset_dir"];
  cout<<"dataset: "<<dataset_dir<<endl;
  string strFile = dataset_dir+"/rgb.txt";
  cout<<"strFile: "<<strFile<<endl;
  string imagepath=dataset_dir+"/rgb";
  cout<<"imagepath: "<<imagepath<<endl;
  ifstream fin(strFile );
  vector<string> vstrImageFilenames;
  vector<double> vTimestamps;
  LoadImages(strFile,vstrImageFilenames,vTimestamps);
  int nImages = vstrImageFilenames.size();
  cout<<"vstrImageFilenames[0]: "<<vstrImageFilenames[0]<<endl;
  //对图像做处理，提取特征，特征匹配得到[R|t]，再做一个重投影的非线性优化
// visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );
    
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 3.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );
    
    
    
    for(int ni=0; ni<nImages; ni++)
    //for(int ni=0; ni<20; ni++)
    {
        // Read image from file
        //im = cv::imread(string(argv[1])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
	im = cv::imread(dataset_dir+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty()){
            cerr << endl << "Failed to load image at: "
                 << dataset_dir << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }
        
        //提取orb特征
        currframe=Frame(im);
	switch (mstate)
	{ 
	  case 	INITSTAL:
	  {
	    //初始化
	    //两帧的匹配点数达到一个阈值才完成初始化
	    if(!initstal)
	    {
	      if(currframe.keypoint_.size()>100)
	      {
		//currframe.Tcw=Eigen::MatrixBase< Derived >::Identity<double>(4,4);
		Eigen::Matrix3d R1=Eigen::Matrix3d::Identity();
		Eigen::Vector3d t1;t1<<0,0,0;
		Sophus::SE3 se3_(R1,t1);
		
		cout<<se3_<<endl;
		
		
		mTcr.block<3,3>(0,0)=se3_.rotation_matrix();
		mTcr.block<3,1>(0,3)=se3_.translation();
		
		currframe.mTcw=lastframe.mTcw=mTcr;
		//currframe.mTcw.block<3,3>(0,0)=se3_.
		lastframe=currframe;
		//initframe.R=Eigen::Matrix3d::Identity();
		//initframe.t=Eigen::Vector3d(0,0,0);
		initstal=true;
	      }
	      //上一步是说第一帧ok了，能够提取足够的特征点，现在开始初始化的第二帧操作
	    }
	    else
	    {
	      if(currframe.keypoint_.size()<100)
	      {
		initstal=false;
		break;
	      }
	      //求去initframe和currframe的运动
	      featurematch();
	      //这是因为匹配点不能太少，不然的话E的结果会出错，没加这句得到的E是 12x3 的
	      if(good_feature_matches.size()<10)break;
	      poseEstimate2d2d();
	      currframe.mTcw=mTcr;
	      Eigen::Matrix3d R1;
	      R1<<currframe.mTcw(0,0),currframe.mTcw(0,1),currframe.mTcw(0,2),currframe.mTcw(1,0),currframe.mTcw(1,1),currframe.mTcw(1,2),currframe.mTcw(2,0),currframe.mTcw(2,1),currframe.mTcw(2,2);
	      Eigen::Vector3d t1;
	      t1<<currframe.mTcw(0,3),currframe.mTcw(1,3),currframe.mTcw(2,3);
	      Sophus::SE3 se3_(R1,t1);
	      cout<<"se3_:                  "<<se3_<<endl;
	      
	      
	      triangulation(lastframe.keypoint_, currframe.keypoint_ ,good_feature_matches, Rcr, tcr, currframe._3dpoint);
	      //triangulation(last_keypoint,curr_keypoint,good_feature_matches,Rcr,tcr,currframe._3dpoint);
	      //TODO	      
	      /*这里把Rt求出之后可以进行迭代优化
	       * 待完成 ceres Eigen 模块
	       * 
	       * 
	       */
	      if(good_feature_matches.size()>7)
		mstate=OK;
	      lastframe=currframe;
	     
	    }
	    break;
	  }
	  case OK:
	  {
	    if(currframe.keypoint_.size()>100)
	    {
	      //cout<<"初始化完成拉～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～"<<endl;
	    //在构造frame的时候就已经提取ORB特征和描述子了，接下来就是curr跟last的特征匹配和恢复深度，3d点，然后也是执行非线性优化
	    featurematch();
	    //这是因为匹配点不能太少，不然的话E的结果会出错，没加这句得到的E是 12x3 的  好的匹配点太少导致帧间匹配精度太差
	    if(good_feature_matches.size()<10)break;
	    
	    poseEstimate3d2d(lastframe._3dpoint,curr_keypoint);
	    
	    //poseEstimate2d2d();
	   
	    //TODO	      
	      /*这里把Rt求出之后可以进行迭代优化
	       * 待完成 ceres Eigen 模块
	       * 
	       * 
	       */
	    currframe.mTcw=mTcr*lastframe.mTcw ;
	    //currframe.Twc=poseEstimate();
	    cout<<currframe.mTcw<<"in OK"<<endl;
	    //currframe.Twc=Tcr_* lastframe.Twc;
	    //currframe.Tcw=lastframe.Tcw*Tcr_ ;
	    
	    triangulation(lastframe.keypoint_,currframe.keypoint_,good_feature_matches,Rcr,tcr,currframe._3dpoint);
	    
	      //这里再加些判断，如果xxx不行，就将num_lost++，等到num_lost到一定数量了，就将 mstate 设为LOST
	    //TODO
	      //这里暂时是每一帧都进行画图操作，可以改进的就是去多线程建图
	    }

	    break;
	  }
	  case LOST:
	  {
	    
	    break;
	  }
	    
	}
	
	//cout<<currframe.Tcw<<"currframe.Tcw before the mapping!!!"<<endl;
	
	// show the map and the camera pose 	
	//Sophus::SE3 Tcw = currframe.Twc.inverse();
	//Sophus::SE3 Tcw = currframe.Tcw;
	auto T=currframe.mTcw.inverse();
	cv::Affine3d MT(
	  cv::Affine3d::Mat3(T(0,0),T(0,1),T(0,2),
			      T(1,0),T(1,1),T(1,2),
			     T(2,0),T(2,1),T(2,2)
	  )
	  ,cv::Affine3d::Vec3(T(0,3),T(1,3),T(2,3)));
	cv::Affine3d M(
	  cv::Affine3d::Mat3(currframe.mTcw(0,0),currframe.mTcw(0,1),currframe.mTcw(0,2),
			      currframe.mTcw(1,0),currframe.mTcw(1,1),currframe.mTcw(1,2),
			     currframe.mTcw(2,0),currframe.mTcw(2,1),currframe.mTcw(2,2)
	  )
	  ,cv::Affine3d::Vec3(currframe.mTcw(0,3),currframe.mTcw(1,3),currframe.mTcw(2,3)));
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
        
        cv::imshow("image", im );
        cv::waitKey(1);
        //vis.setWidgetPose( "Camera", M);
	vis.setWidgetPose( "Camera", MT);
        vis.spinOnce(1, false);
	
    }
}






























//#优化位姿
//输入：ref中3D点，curr中对应像素点（2D），两帧相对位姿Tcr

Eigen::Matrix<double,2,6> Jakobi_T(Eigen::Vector3d point_3d)
{
  int fx=520.9 , cx=325.1, fy=521.0, cy=249.7;
  Eigen::Matrix<double,2,6> Jkb;
  Jkb(0,0)=fx/point_3d[2];
  Jkb(0,1)=0;
  Jkb(0,2)=-fx*point_3d[0]/(point_3d[2]*point_3d[2]);
  Jkb(0,3)=-fx*point_3d[0]*point_3d[1]/(point_3d[2]*point_3d[2]);
  Jkb(0,4)=fx+fx*(point_3d[0]*point_3d[0])/(point_3d[2]*point_3d[2]);
  Jkb(0,5)=-fx*point_3d[1]/point_3d[2];
  Jkb(1,0)=0;
  Jkb(1,1)=fy/point_3d[2];
  Jkb(1,2)=-fy*point_3d[1]/(point_3d[2]*point_3d[2]);
  Jkb(1,3)=-fy-fy*(point_3d[1]*point_3d[1])/(point_3d[2]*point_3d[2]);
  Jkb(1,4)=fy*point_3d[0]*point_3d[1]/(point_3d[2]*point_3d[2]);
  Jkb(1,5)=fy*point_3d[0]/point_3d[2];
  return -Jkb;
}
Sophus::SE3 EigentoSE3(Eigen::MatrixXd Tcr)
{
  Eigen::Matrix3d R=Tcr.block<3,3>(0,0);
  Eigen::Vector3d t=Tcr.block<3,1>(0,3);
  Sophus::SE3 se3(R,t);
  return se3;
}
void Eigen_BA_onlyPose(int N,vector<cv::Point3d> ref_3D,vector<cv::Point2d> curr_2d,Eigen::Matrix4d &Tcr)
{
  Eigen::Matrix3d K;
  K<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
  Eigen::Vector2d residual;
  Eigen::Vector2d residual1;
  Eigen::Vector3d curr_3D;
  Eigen::Matrix<double,6,6> Hessin;
  Eigen::Matrix<double,2,6> J;
  Eigen::Matrix<double,6,1> g;//g=-JT*f(x)
  Eigen::Matrix<double,6,1> dx;
  Sophus::SE3 se3,se3_update;
  
  for(int i=0;i<ref_3D.size();i++)
  {
    //curr_3D=Tcr*ref_3D
    Eigen::Vector3d P3D;P3D<<ref_3D[i].x,ref_3D[i].y,ref_3D[i].z;
    Eigen::Matrix3d Rcr;Rcr=Tcr.block<3,3>(0,0);
    Eigen::Vector3d tcr;tcr=Tcr.block<3,1>(0,3);
    //curr_3D=Tcr*P3D;4×4 3×1
    curr_3D=Rcr*P3D+tcr;
    Eigen::Vector3d curr_3D_norm;
    curr_3D_norm<< curr_3D[0]/curr_3D[2],curr_3D[1]/curr_3D[2],1;
    Eigen::Vector3d norm_=K*curr_3D_norm;
    Eigen::Vector2d curr_2d_(norm_(0),norm_(1));
    Eigen::Vector2d P2D;
    P2D<<curr_2d[i].x,curr_2d[i].y;
    residual=curr_2d_-P2D;
    //residual(1)=curr_2d_(1)-curr_2d.y();
    //雅克比应该是每一次点都求解一次，或者说每个误差项求解一个对应的雅克比
    J=Jakobi_T(curr_3D);
    Hessin=J.transpose()*J;
    g=-J.transpose()*residual;
    //旋转在前 平移在后
    dx=Hessin.inverse()*g;
    if(dx.norm()<0.001)break;
    if(i!=0 && residual.norm()>residual1.norm())break;
    //------------------------------------------------------------update Pose
    se3=EigentoSE3(Tcr);
    se3_update=Sophus::SE3::exp(dx)*se3;
    Tcr.block<3,3>(0,0) = se3_update.rotation_matrix();
    Tcr.block<3,1>(0,3) = se3_update.translation();
    //------------------------------------------------------------
    
    residual1=residual;
  }
  
  
} 



//#优化位姿与3D点
//输入：ref中3D点，curr中对应像素点（2D），两帧相对位姿Tcr


Eigen::MatrixXd Jakobi_Pose_Point(Eigen::Vector3d point_3d,Eigen::MatrixXd Tcr)
{
  int fx=520.9 , cx=325.1, fy=521.0, cy=249.7;
  Eigen::MatrixXd Jkb;
  Eigen::Matrix<double,2,3> dedp;
  Eigen::Matrix<double,2,3> dedP;
  dedp(0,0)=fx/point_3d[2];
  dedp(0,1)=0;
  dedp(0,2)=-fx*point_3d[0]/(point_3d[2]*point_3d[2]);
  dedp(1,0)=0;
  dedp(1,1)=fy/point_3d[2];
  dedp(1,2)=-fy*point_3d[1]/(point_3d[2]*point_3d[2]);
  Eigen::Matrix3d R=Tcr.block<3,3>(0,0);
  dedP=dedp*R;
  
  Jkb(0,0)=fx/point_3d[2];
  Jkb(0,1)=0;
  Jkb(0,2)=-fx*point_3d[0]/(point_3d[2]*point_3d[2]);
  Jkb(0,3)=-fx*point_3d[0]*point_3d[1]/(point_3d[2]*point_3d[2]);
  Jkb(0,4)=fx+fx*(point_3d[0]*point_3d[0])/(point_3d[2]*point_3d[2]);
  Jkb(0,5)=-fx*point_3d[1]/point_3d[2];
  Jkb(0,6)=dedP(0,0);
  Jkb(0,7)=dedP(0,1);
  Jkb(0,8)=dedP(0,2);
 
  Jkb(1,0)=0;
  Jkb(1,1)=fy/point_3d[2];
  Jkb(1,2)=-fy*point_3d[1]/(point_3d[2]*point_3d[2]);
  Jkb(1,3)=-fy-fy*(point_3d[1]*point_3d[1])/(point_3d[2]*point_3d[2]);
  Jkb(1,4)=fy*point_3d[0]*point_3d[1]/(point_3d[2]*point_3d[2]);
  Jkb(1,5)=fy*point_3d[0]/point_3d[2];
  Jkb(1,6)=dedP(1,0);
  Jkb(1,7)=dedP(1,1);
  Jkb(1,8)=dedP(1,2);
  return -Jkb;
}
void Eigen_BA_Pose_and_Point(int N,vector<cv::Point3d> ref_3D,vector<cv::Point2d> curr_2d,Eigen::MatrixXd &Tcr)
{
  Eigen::Matrix3d K;K<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
  Eigen::Vector2d residual;
  Eigen::Vector2d residual1;
  Eigen::Vector3d curr_3D;
  Eigen::Matrix<double,9,9> Hessin;
  Eigen::Matrix<double,2,9> J;
  Eigen::Matrix<double,9,1> g;//g=-JT*f(x)
  Eigen::Matrix<double,9,1> dx;
  Eigen::Matrix<double,6,1> dx_Pose;
  Eigen::Matrix<double,3,1> dx_Point;
  Sophus::SE3 se3,se3_update;
  
  for(int i=0;i<ref_3D.size();i++)
  {
    //curr_3D=Tcr*ref_3D
    Eigen::Vector3d P3D;
    P3D<<ref_3D[i].x,ref_3D[i].y,ref_3D[i].z;
    Eigen::Matrix3d Rcr;Rcr=Tcr.block<3,3>(0,0);
    Eigen::Vector3d tcr;tcr=Tcr.block<3,1>(0,3);
    //curr_3D=Tcr*P3D;4×4 3×1
    curr_3D=Rcr*P3D+tcr;
    Eigen::Vector3d curr_3D_norm;
    curr_3D_norm<< curr_3D[0]/curr_3D[2],curr_3D[1]/curr_3D[2],1;
    Eigen::Vector3d norm_=K*curr_3D_norm;
    Eigen::Vector2d curr_2d_(norm_[0],norm_[1]);
    Eigen::Vector2d P2D;
    P2D<<curr_2d[i].x,curr_2d[i].y;
    residual=curr_2d_-P2D;
   // residual(1)=curr_2d_(1)-curr_2d(1);
    //此时dx是2X9
    J=Jakobi_Pose_Point(curr_3D,Tcr);
    Hessin=J.transpose()*J;
    g=-J.transpose()*residual;
    //旋转在前 平移在后
    dx=Hessin.inverse()*g;
    if(dx.norm()<0.001)break;
    if(i!=0 && residual.norm()>residual1.norm())break;
    //------------------------------------------------------------update Pose
    se3=EigentoSE3(Tcr);
    dx_Pose<<dx(0),dx(1),dx(2),dx(3),dx(4),dx(5);
    dx_Point<<dx(6),dx(7),dx(8);
    se3_update=Sophus::SE3::exp(dx_Pose)*se3;
    Tcr.block<3,3>(0,0) = se3_update.rotation_matrix();
    Tcr.block<3,1>(0,3) = se3_update.translation();
    //------------------------------------------------------------updata Point
    ref_3D[i].x+=dx_Point(0);
    ref_3D[i].y+=dx_Point(1);
    ref_3D[i].z+=dx_Point(2);
    //------------------------------------------------------------
    residual1=residual;
  }
  
  
} 






















