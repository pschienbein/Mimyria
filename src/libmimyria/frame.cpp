#include "pch.hpp"
#include "frame.hpp"

using namespace std;

Frame::~Frame()
{
}

void Frame::add(FramePtr pOther)
{
    // move the array over without copying it
    if(!m_mPGT && pOther->m_mPGT)
        m_mPGT = std::exchange(pOther->m_mPGT, std::nullopt);

    if(m_mAPT.empty() && !pOther->m_mAPT.empty())
        m_mAPT = std::move(pOther->m_mAPT);

    if(m_sSymbols.empty() && !pOther->m_sSymbols.empty())
        m_sSymbols = std::move(pOther->m_sSymbols);

    if(m_vPositions.empty() && !pOther->m_vPositions.empty())
        m_vPositions = std::move(pOther->m_vPositions);

    if(m_vVelocities.empty() && !pOther->m_vVelocities.empty())
        m_vVelocities = std::move(pOther->m_vVelocities);

    if(!m_vDipole && pOther->m_vDipole)
        m_vDipole = pOther->m_vDipole;
    if(!m_mPolarizability && pOther->m_mPolarizability)
        m_mPolarizability = pOther->m_mPolarizability;
}
