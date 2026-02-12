#ifndef __TASK_HPP__
#define __TASK_HPP__

#include "factory.hpp"
#include "frame.hpp"
#include "configfile.hpp"

class Task 
{
public:
    virtual ~Task() = 0;

    virtual void setup(ConfigFilePtr, ConfigFile::SectionPtr);

    virtual void onFrame(FramePtr pFrame);
    virtual void onTrajectoryOpened();
    virtual void onTrajectoryFinished();
    virtual void onFinished();
};

typedef std::shared_ptr<Task> TaskPtr;

namespace registry
{
    typedef Factory<Task> TaskFactory;
}

#endif 
