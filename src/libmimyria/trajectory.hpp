#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include "factory.hpp"
#include "frame.hpp"

class Trajectory
{
public:
    typedef std::map<std::string, std::string> KVPairs;

    Trajectory();
    virtual ~Trajectory() = 0;

    // close trajectory
    virtual void close();

    // open a file
    virtual void open(const std::string& sFilename, const KVPairs& extra= KVPairs());
    // open a c++ stream
    virtual void open(std::istream* pisInput, const std::string& sName = "", const KVPairs& extra = KVPairs());

    // read next frame
    virtual FramePtr readNextFrame() = 0;

    // get the given name
    inline const std::string& getName() const { return m_sName; }

    // ESTIMATE the number of frames contained in the opened stream or file. 
    // not 100 accurate, but gives a rough estimation which is surprisingly accurate for XYZ files.
    virtual size_t estimateTotalNumberOfFrames();

protected:
    std::istream* m_pisInput; 
    std::ifstream m_ifsInFile;

private:
    std::string m_sName;
};

typedef std::shared_ptr<Trajectory> TrajectoryPtr;
typedef std::shared_ptr<const Trajectory> ConstTrajectoryPtr;

namespace registry
{
    typedef Factory<Trajectory> TrajectoryFactory;
}

#endif
