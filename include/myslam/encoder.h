#ifndef ENCODER_H
#define ENCODER_H

namespace myslam
{

class Encoder
{
public:
    Encoder(): enl_(0.0), enr_(0.0), timestamp_(0.0){}
    Encoder(const double& enl, const double& enr, const double& timestamp):
    enl_(enl), enr_(enr), timestamp_(timestamp) {}
    
    double enl_, enr_, timestamp_;
}; // class Encoder

} // namespace dre_slam

#endif