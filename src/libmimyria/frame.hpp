#ifndef __FRAME_HPP__
#define __FRAME_HPP__

#include "box.hpp"
#include <memory>

typedef std::array<Eigen::Matrix3d, 3> Tensor3d; 

class Frame
{
public:
    typedef std::shared_ptr<Frame> FramePtr;

    ~Frame();

    // this operation adds all information from the other frame to this one, 
    // without overwriting anything. 
    // This means only ADDITIONAL information is added
    void add(FramePtr pOther);

    std::vector<std::string> m_sSymbols;
    std::vector<Eigen::Vector3d> m_vPositions;
    std::vector<Eigen::Vector3d> m_vVelocities;
    std::vector<Eigen::Matrix3d> m_mAPT;
    std::optional<std::vector<Tensor3d>> m_mPGT;

    // ... as product of APT times velocity
    std::optional<std::vector<Eigen::Vector3d>> m_vAtomDipoles;

    std::optional<Eigen::Vector3d> m_vDipole;
    std::optional<Eigen::Matrix3d> m_mPolarizability;

    BoxPtr m_pBox;
};

//using FramePtr = std::shared_ptr<Frame>;
typedef Frame::FramePtr FramePtr;

#endif

