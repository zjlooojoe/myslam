#include <encoder_integration.h>
#include <sophus/se2.h>
#include <sophus/se3.h>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>


namespace myslam
{

EncoderIntegration::EncoderIntegration ( const double& kl, const double& kr, const double& b, const double& k ) :
    kl_ ( kl ), kr_ ( kr ), b_ ( b ), k_ ( k ), is_init_ ( false ), x_ ( 0.0 ), y_ ( 0.0 ), th_ ( 0.0 )
{
}

Sophus::SE2 EncoderIntegration::addEncoder ( const Encoder& enc )
{

    if ( is_init_ == false ) 
    {
        last_enc_ = enc;
        Trr_ = Sophus::SE2();
        cov_.setZero();
        is_init_ = true;
    } 
    else 
    { 
        double delta_enl = enc.enl_ - last_enc_.enl_;
        double delta_enr = enc.enr_ - last_enc_.enr_;
        last_enc_ = enc;
		
        double delta_sl = kl_ * delta_enl;
        double delta_sr = kr_ * delta_enr;

        double delta_theta = ( delta_sr - delta_sl ) / b_;
        double delta_s = 0.5 * ( delta_sr + delta_sl );

        double tmp_th = th_ + 0.5 * delta_theta;
        double cos_tmp_th = cos ( tmp_th );
        double sin_tmp_th = sin ( tmp_th );

        x_ += delta_s * cos_tmp_th;
        y_ += delta_s * sin_tmp_th;
        th_ += delta_theta;
        th_ = normAngle ( th_ ); 

        Trr_ = Sophus::SE2 ( th_, Eigen::Vector2d ( x_, y_ ) );

        Eigen::Matrix3d Gt;//(5.10)
        Gt << 1.0, 0.0, -delta_s * sin_tmp_th,
           0.0, 1.0, delta_s * cos_tmp_th,
           0.0, 0.0, 1.0;

        Eigen::Matrix<double, 3, 2> Gu;//(5.11)
        Gu << 0.5  * ( cos_tmp_th - delta_s * sin_tmp_th / b_ ), 0.5  * ( cos_tmp_th + delta_s * sin_tmp_th / b_ ),
           0.5  * ( sin_tmp_th + delta_s * cos_tmp_th /b_ ), 0.5  * ( sin_tmp_th - delta_s * cos_tmp_th/b_ ),
           1.0/b_, -1.0/b_;

        Eigen::Matrix2d sigma_u;
        sigma_u << k_ * k_ * delta_sr * delta_sr, 0.0, 0.0, k_ * k_* delta_sl * delta_sl;

	//(5.9)
        cov_ = Gt * cov_ *Gt.transpose() +  Gu * sigma_u * Gu.transpose() ;
    } // odom

    return Trr_;
}

Sophus::SE2 EncoderIntegration::addEncoders ( const std::vector<Encoder>& encs )
{
    for ( size_t i = 0; i < encs.size(); i ++ ) {
        const Encoder& enc = encs.at ( i );
        addEncoder ( enc );
    }
}


Eigen::Matrix3d EncoderIntegration::getCov()
{
    return cov_;
} // getCov


Sophus::SE2 EncoderIntegration::getTrr()
{
    return Trr_;
} // getTrr

//common.cpp
double EncoderIntegration::normAngle ( double angle )
{
    static double Two_PI = 2.0 * M_PI;
    
    if ( angle >= M_PI ) {
        angle -= Two_PI;
    }
    if ( angle < -M_PI ) {
        angle += Two_PI;
    }
    return angle;
}


int EncoderIntegration::DescriptorDistance ( const cv::Mat &a, const cv::Mat &b )
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for ( int i=0; i<8; i++, pa++, pb++ ) {
        unsigned  int v = *pa ^ *pb;
        v = v - ( ( v >> 1 ) & 0x55555555 );
        v = ( v & 0x33333333 ) + ( ( v >> 2 ) & 0x33333333 );
        dist += ( ( ( v + ( v >> 4 ) ) & 0xF0F0F0F ) * 0x1010101 ) >> 24;
    }
    return dist;
}

std::vector<cv::Mat> EncoderIntegration::CvMat2DescriptorVector ( const cv::Mat &Descriptors )
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve ( Descriptors.rows );
    for ( int j=0; j<Descriptors.rows; j++ ) {
        vDesc.push_back ( Descriptors.row ( j ) );
    }

    return vDesc;
}

Eigen::Matrix4d EncoderIntegration::AngleAxisTrans2EigenT ( cv::Vec3d rvec, cv::Vec3d tvec )
{
    /* convert rotation angle to R matrix */
    cv::Mat R;
    cv::Rodrigues ( rvec, R );

    /* convert to eigen style */
    Eigen::Matrix4d T;
    T<<
     R.at<double> ( 0, 0 ), R.at<double> ( 0, 1 ), R.at<double> ( 0, 2 ), tvec[0],
          R.at<double> ( 1, 0 ), R.at<double> ( 1, 1 ), R.at<double> ( 1, 2 ), tvec[1],
          R.at<double> ( 2, 0 ), R.at<double> ( 2, 1 ), R.at<double> ( 2, 2 ), tvec[2],
          0.,0.,0.,1.;
    return T;
}

Sophus::SE2 EncoderIntegration::EigenT2Pose2d ( Eigen::Matrix4d& T )
{
    double theta = atan2 ( T ( 1,0 ), T ( 0,0 ) );
    double x = T ( 0, 3 );
    double y = T ( 1, 3 );
    return Sophus::SE2 ( theta, Eigen::Vector2d ( x, y ) );
}

} // namespace dre_slam
