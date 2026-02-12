#include "pch.hpp"
#include <algorithm>
#include "trajectory.hpp"

using namespace std;

Trajectory::Trajectory()
{}

Trajectory::~Trajectory()
{}

void Trajectory::close()
{
    if(m_ifsInFile.is_open())
        m_ifsInFile.close();
}

void Trajectory::open(const string& sFilename, const KVPairs& extra)
{
    m_ifsInFile.open(sFilename);
    if(!m_ifsInFile.good())
        THROW(runtime_error, "Could not open file \"", sFilename, "\"");
    open(&m_ifsInFile, sFilename, extra);
}

void Trajectory::open(istream* pisInput, const string& sName, const KVPairs&)
{
    m_pisInput = pisInput;
    m_sName = sName;
}

size_t Trajectory::estimateTotalNumberOfFrames()
{
    return 0;
}
