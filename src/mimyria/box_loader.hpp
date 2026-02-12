#include "configfile.hpp"
#include "box.hpp"
#include "frame.hpp"

class BoxLoader
{
public:
    BoxPtr get(FramePtr pFrame);
    void nextTrajectory();
    void setup(ConfigFilePtr pCF);

private:
    void read_mode(ConfigFilePtr pCF, ConfigFile::SectionPtr pCell);

    enum class Mode
    {
        None,
        Constant, 
        Trajectory, 
        Frame
    }
    m_mode{Mode::None};

    std::ifstream m_ifsCell;
    BoxPtr m_pConstantBox;
};
