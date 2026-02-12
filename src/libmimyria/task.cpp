#include "pch.hpp"
#include "task.hpp"

using namespace std;

Task::~Task()
{
}

void Task::setup(ConfigFilePtr, ConfigFile::SectionPtr)
{
}

void Task::onFrame(FramePtr pFrame)
{
}

void Task::onTrajectoryOpened()
{
}

void Task::onTrajectoryFinished()
{
}

void Task::onFinished()
{
}
