
#ifndef ENCODER_INTEGRATION_H
#define ENCODER_INTEGRATION_H

#include <sophus/se2.h>
#include <encoder.h>

namespace myslam
{
class EncoderIntegration
{
public:

    EncoderIntegration(){}
    EncoderIntegration(const double& kl, const double& kr, const double& b, const double& k);
    Sophus::SE2 addEncoder(const Encoder& enc); // 
    Sophus::SE2 addEncoders(const std::vector< Encoder >& encs);

    //common.cpp
    Sophus::SE2 getTrr();
    Eigen::Matrix3d getCov();
    double normAngle ( double angle );
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    std::vector<cv::Mat> CvMat2DescriptorVector(const cv::Mat &Descriptors);
    Eigen::Matrix4d AngleAxisTrans2EigenT(cv::Vec3d rvec, cv::Vec3d tvec);
    Sophus::SE2 EigenT2Pose2d(Eigen::Matrix4d& T);

private:
    double kl_, kr_, b_, k_;

    Sophus::SE2 Trr_;
    Eigen::Matrix3d cov_;

    bool is_init_;
    Encoder last_enc_;
    double x_, y_, th_;
    
}; // class EncoderIntegratation

} // namespace dre_slam

#endif