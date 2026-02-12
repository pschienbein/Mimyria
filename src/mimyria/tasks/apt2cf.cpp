#include "../pch.hpp"

#include "task.hpp"
#include "correlation.hpp"

using namespace std;

class APT2CF : public Task
{
public:
    APT2CF() 
        : m_nTrajectories(0)
    {
    }

    virtual void onFrame(FramePtr pFrame) override
    {
        if(pFrame->m_mAPT.size() != pFrame->m_vVelocities.size() || pFrame->m_mAPT.empty())
            THROW(runtime_error, "To calculate correlation function from APTs, velocities and APTs need to be available at every frame!");

        const size_t N = pFrame->m_mAPT.size(); 
        Vector3d Mdot = Vector3d::Zero();

        for(size_t i = 0; i < N; ++i)
            Mdot += pFrame->m_mAPT[i] * pFrame->m_vVelocities[i];

        for(size_t k = 0; k < 3; ++k)
            m_aMdot[k].push_back(Mdot(k));
    }

    virtual void setup(ConfigFilePtr pCF, ConfigFile::SectionPtr pSec) override
    {
        // get out file name
        m_sOutFn = pCF->get(pSec, "cf_out");
        cerr << "# Writing correlation function to: " << m_sOutFn << endl;
    }

    virtual void onTrajectoryFinished() override
    {
        // now compute the correlation functions
        const auto N = m_aMdot[0].size();

        for(size_t k = 0; k < 3; ++k)
        {
            m_aadCorrFunc[k].commit(m_aMdot[k].data(), N);
            m_aadCorrFunc[k].finishTrajectory();

            // reset buffer
            m_aMdot[k].clear();
        }

        ++m_nTrajectories;

        // store the correlation function
        write_cf();
    }

private:
    void write_cf()
    {
        // write to file
        ofstream ofs(m_sOutFn, ios::trunc);
        ofs << "# Correlation function after " << m_nTrajectories << " trajectories\n";
        ofs << "# Time | AC(x) | AC(y) | AC(z)\n";
        for(size_t i = 0; i < m_aadCorrFunc[0].getNumFrames(); ++i)
        {
            ofs << i << " ";
            for(size_t k = 0; k < 3; ++k)
                ofs << m_aadCorrFunc[k][i] << " ";
            ofs << "\n";
        }
    }

    // per trajectory storage for the dipole moments
    std::vector<double> m_aMdot[3];

    // correlator
    AutoCorrelation m_aadCorrFunc[3];

    // output file name
    std::string m_sOutFn;

    // number of processed trajectories
    size_t m_nTrajectories;
};

REGISTER(registry::TaskFactory, "apt2cf", APT2CF);
