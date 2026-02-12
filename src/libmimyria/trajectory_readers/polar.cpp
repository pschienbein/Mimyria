#include "../pch.hpp"
#include "../exception.hpp"
#include "../trajectory.hpp"
#include "Eigen/src/Core/Matrix.h"

using namespace std;

class ListOfPolarizabilities : public Trajectory 
{
public:
    ListOfPolarizabilities()
    { 
    }
    virtual ~ListOfPolarizabilities()
    { 
        close();
    }

    // read next frame
    virtual FramePtr readNextFrame() 
    {
        if(m_pisInput->eof())
            return nullptr;

        string line; 
        bool bComment = false;
        do 
        {
            getline(*m_pisInput, line);
            bComment = line.find("#") != string::npos;
        }
        while(bComment);

        FramePtr pFrame = make_shared<Frame>();
        pFrame->m_mPolarizability.emplace();

        istringstream reader(line);
        string unused;
        reader >> unused >> unused;
        for(size_t i = 0; i < 3; ++i)
            for(size_t j = 0; j < 3; ++j)
                reader >> (*pFrame->m_mPolarizability)(i, j);

        // happens when there's an empty line at the end of the file
        if(reader.fail())
            return nullptr;

        return pFrame;
    }

    // ESTIMATE the number of frames contained in the opened stream or file. 
    // not 100 accurate, but gives a rough estimation which is surprisingly accurate for XYZ files.
    virtual size_t estimateTotalNumberOfFrames()
    {
    }

private: 
};


REGISTER(registry::TrajectoryFactory, "polar", ListOfPolarizabilities);
