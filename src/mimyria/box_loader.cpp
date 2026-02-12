#include "pch.hpp"
#include <exception>
#include "box_loader.hpp"

using namespace std;

// valid options for the control file are
// 1. constant box:
// a1=
// a2=
// a3=
// b1=
// b2=
// b3=
// c1=
// c2=
// c3=
//
// 2. file that contains lattice vectors
// cellfn=<path to file>; one line contains a1 a2 a3 b1 b2 b3 c1 c2 c3
// mode=constant -> expect a single box
// mode=trajectory -> expect one line for each trajectory
// mode=frame -> expect one line for each frame
//
// 3. lattice contained in the trajectory file along with each frame
// mode=frame

BoxPtr BoxLoader::get(FramePtr pFrame)
{
    if(m_mode == Mode::Frame && !m_ifsCell.is_open())
        THROW(runtime_error, "[cell]: Reading lattice from trajectory file not yet implemented!");

    if(m_pConstantBox)
        return make_shared<Box>(*m_pConstantBox);

    return nullptr;
}

void BoxLoader::nextTrajectory() 
{
    if(m_mode == Mode::Trajectory && m_ifsCell.is_open())
    {
        // read one line
        string sLine;
        getline(m_ifsCell, sLine);

        if(m_ifsCell.fail())
            THROW(runtime_error, "[cell]: Failed to read next line from cell file");

        try {
            istringstream line(sLine);
            line.exceptions(ios::failbit | ios::badbit);

            // read lattice constants from line:
            double a1 = 0, a2 = 0, a3 = 0, b1 = 0, b2 = 0, b3 = 0, c1 = 0, c2 = 0, c3 = 0;
            line >> a1 >> a2 >> a3 >> b1 >> b2 >> b3 >> c1 >> c2 >> c3;

            m_pConstantBox = make_shared<Box>();
            Vector3d A(a1, a2, a3), B(b1, b2, b3), C(c1, c2, c3);
            m_pConstantBox->setup(A, B, C);
            //m_pConstantBox->setup(Vector3d(a1, a2, a3), Vector3d(b1, b2, b3), Vector3d(c1, c2, c3));

            cerr << "[cell]: Now loaded cell lattice:\n"
                 << "[cell]: A = " << A.transpose() << "\n"
                 << "[cell]: B = " << B.transpose() << "\n"
                 << "[cell]: C = " << C.transpose() << "\n";
        }
        catch(ios_base::failure& e)
        {
            THROW_WITH_NESTED(runtime_error, "[cell]: Failed to parse line \"", sLine, "\" in given cell file");
        }
    }
}

void BoxLoader::read_mode(ConfigFilePtr pConfig, ConfigFile::SectionPtr pCell)
{
    string sMode = pConfig->get(pCell, "mode");

    if(sMode == "trajectory")
        m_mode = Mode::Trajectory;
    else if (sMode == "none")
        m_mode = Mode::None;
    else if (sMode == "constant")
        m_mode = Mode::Constant;
    else if (sMode == "frame")
        m_mode = Mode::Frame;
    else
        THROW(runtime_error, "[cell]: Unknown mode=", sMode, " valid options are none, constant, trajectory, and frame");
}

void BoxLoader::setup(ConfigFilePtr pConfig)
{
    BoxPtr pBox = nullptr;
    if(pConfig->isSectionPresent("cell"))
    {
        auto pCell = pConfig->getSection("cell");

        // is there a file that can be opened
        if(pConfig->isKeyPresent(pCell, "cellfn"))
        {
            string sFn = pConfig->get(pCell, "cellfn");
            try {
                read_mode(pConfig, pCell);
            }
            catch(runtime_error& e)
            {
                THROW_WITH_NESTED(runtime_error, "[cell]: When providing a cellfn, mode= must be set (constant, trajectory, frame)");
            }
            m_ifsCell.open(sFn);
            if(!m_ifsCell.good())
                THROW(runtime_error, "Could not open \"", sFn, "\"");
        }
        else
        {
            // is there a1?
            if(pConfig->isKeyPresent(pCell, "a1"))
            {
                // ... if yes, expect the others and define a constant cell!
                try {
                    m_pConstantBox = make_shared<Box>();
                    m_pConstantBox->setup(Vector3d(
                                pConfig->as<double>(pCell, "a1"),
                                pConfig->as<double>(pCell, "a2"),
                                pConfig->as<double>(pCell, "a3")
                                ),
                            Vector3d(
                                pConfig->as<double>(pCell, "b1"),
                                pConfig->as<double>(pCell, "b2"),
                                pConfig->as<double>(pCell, "b3")
                                ),
                            Vector3d(
                                pConfig->as<double>(pCell, "c1"),
                                pConfig->as<double>(pCell, "c2"),
                                pConfig->as<double>(pCell, "c3")
                                )
                         );
                }
                catch(exception& e)
                {
                    THROW_WITH_NESTED(runtime_error, "[cell]: One of a1, a2, a3, b1, b2, b3, c1, c2, c3 could not be read");
                }

                // setup / as throws if not successful
                m_mode = Mode::Constant;
            }
            else
            {
                // read the mode
                try {
                    read_mode(pConfig, pCell);
                }
                catch(exception&)
                {
                    THROW_WITH_NESTED(runtime_error, "[cell]: Got a [cell] section, but no cellfn, no constant cell definition, and no mode is provided. Either remove [cell] or set mode=none (no cell) or mode=frame (read from trajectory)");
                }

                // now only None and Frame is a valid option!
                if(m_mode != Mode::Frame && m_mode != Mode::None)
                    THROW(runtime_error, "[cell]: When no cell definition is given, only mode=none (no cell) or mode=frame (read from trajectory) are possible options!");
            }
        }
    }
};
