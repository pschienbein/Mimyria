#include "../pch.hpp"

#include "task.hpp"
#include "correlation.hpp"

using namespace std;

class PGT2CF : public Task
{
public:
    PGT2CF()
    {
    }

    virtual void onFrame(FramePtr pFrame) override
    {
        if(!pFrame->m_mPGT || pFrame->m_mPGT->size() != pFrame->m_vVelocities.size())
            THROW(runtime_error, "To calculate correlation functions from PGTs, velocities and PGTs must be available at every frame");

        const size_t N = pFrame->m_mPGT->size();
        Matrix3d aDot = Matrix3d::Zero();

        for(size_t i = 0; i < N; ++i)
        {
            for(size_t k = 0; k < 3; ++k)
                aDot += (*pFrame->m_mPGT)[i][k] * pFrame->m_vVelocities[i][k];
        }

        // copy & symmetrize polarizability
        for(size_t k = 0; k < NumIndices; ++k)
        {
            const size_t i = Indices[k].first;
            const size_t j = Indices[k].second;

            const double dThis = 0.5 * (aDot(i,j) + aDot(j,i));
            m_aData[k].push_back(dThis);
        }
    }

    virtual void setup(ConfigFilePtr pCF, ConfigFile::SectionPtr pSec) override
    {
        // get out file name
        m_sOutFn = pCF->get(pSec, "cf_out");
        cerr << "# Writing correlation function to: " << m_sOutFn << endl;
    }

    virtual void onTrajectoryFinished() override
    {
        const auto N = m_aData[0].size();
        vector<double> buffer(N);

        for(size_t k = 0; k < NumCC; ++k)
        {
            const size_t k1 = CCIndices[k].first;
            const size_t k2 = CCIndices[k].second;

            m_aadCorrFuncCC[k].commit(m_aData[k1].data(), m_aData[k2].data(), N);
            m_aadCorrFuncCC[k].finishTrajectory();
        }

        for(size_t k = 0; k < NumIndices; ++k)
        {
            m_aadCorrFunc[k].commit(m_aData[k].data(), N);
            m_aadCorrFunc[k].finishTrajectory();

            m_aData[k].clear();
        }

        ++m_nTrajectories;

        // store the correlation function
        store();
    }

private:
    void store()
    {
        // print all trajectories to file
        ofstream ofs(m_sOutFn, ios::trunc);

        // create header
        ofs << "# Correlation function after " << m_nTrajectories << " trajectories\n";
        ofs << "# Time | ";
        for(size_t k = 0; k < NumIndices; ++k)
            ofs << "AC(" << IndexNames[k] << ") | ";
        for(size_t k = 0; k < NumCC; ++k)
        {
            const size_t k1 = CCIndices[k].first;
            const size_t k2 = CCIndices[k].second;
            ofs << "CC(" << IndexNames[k1] << "," << IndexNames[k2] << ") | ";
        }
        ofs << "\n";

        for (size_t t = 0; t < m_aadCorrFunc[0].getNumFrames(); ++t)
        {
            ofs << t << " ";
            for(size_t k = 0; k < NumIndices; ++k)
                ofs << m_aadCorrFunc[k][t] << " ";
            for(size_t k = 0; k < NumCC; ++k)
                ofs << m_aadCorrFuncCC[k][t] << " ";
            ofs << "\n";
        }
    }

    constexpr static size_t NumIndices = 6;
    constexpr static std::array<std::pair<size_t, size_t>, NumIndices> Indices = {{ {0,0}, {1,1}, {2,2}, {0,1}, {0,2}, {1,2} }};
    constexpr static std::array<const char*, NumIndices> IndexNames = {{ "xx", "yy", "zz", "xy", "xz", "yz" }};
    std::array<std::vector<double>, NumIndices> m_aData;
    std::array<AutoCorrelation, NumIndices> m_aadCorrFunc;

    constexpr static size_t NumCC = 15;
    constexpr static std::array<std::pair<size_t, size_t>, NumCC> CCIndices = {{ {0,1}, {0,2}, {0,3}, {0,4}, {0,5}, {1,2}, {1,3}, {1,4}, {1,5}, {2,3}, {2,4}, {2,5}, {3,4}, {3,5}, {4,5} }};
    std::array<CrossCorrelation, NumCC> m_aadCorrFuncCC;

    string m_sOutFn;

    size_t m_nTrajectories{0};
};

REGISTER(registry::TaskFactory, "pgt2cf", PGT2CF);
