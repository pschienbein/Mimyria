#include "pch.hpp"
#include "correlation.hpp"
#include "exception.hpp"

#include <fftw3.h>
#include <cwctype>
#include <numeric>

using namespace std;

Correlation::Correlation(unique_ptr<correlator::Correlator> pTCFDriver)
   : m_pCorrelator(move(pTCFDriver))
   , m_type(Type::Undefined)
{
    if(!m_pCorrelator)
        m_pCorrelator = make_unique<CorrelatorFFT>();
}  

Correlation::~Correlation()
{
}

void Correlation::commit(const double* pBuffer, size_t nFrames, vector<double>* paThisOutput)
{
    if(m_type == Type::CC)
        THROW(runtime_error, "Correlation used for cross correlations is now to do a auto correlation, this is not supported...");
    m_type = Type::AC;

    if(m_nAllocatedPlanSize == 0)
    {
        m_aTrajectoryAC.resize(nFrames);
        m_aTotalAC.resize(nFrames);

        m_nAllocatedPlanSize = nFrames;                
    }

    if(nFrames > m_nAllocatedPlanSize)
    {
        stringstream msg;
        msg << "Correlation: Attempting to do FFT length "
            << nFrames 
            << ", but the maximum allowed length is " 
            << m_nAllocatedPlanSize;
        THROW(out_of_range, msg.str());
    }

    m_pCorrelator->correlate(pBuffer, nFrames);
    for(size_t i = 0; i < nFrames; ++i)
        m_aTrajectoryAC[i] += m_pCorrelator->at(i);

    // if requested, also give the data to the optional array returning the immediate AC
    if(paThisOutput)
    {
        paThisOutput->resize(nFrames);
        for(size_t i = 0; i < nFrames; ++i)
            paThisOutput->at(i) = m_pCorrelator->at(i);
    }

    ++m_nTrajectoryCommits;
}

void Correlation::commit(const double* pBufferA, const double* pBufferB, size_t nFrames, vector<double>* paThisOutput)
{
    if(m_type == Type::AC)
        THROW(runtime_error, "Correlation used for auto correlations is now to do a cross correlation, this is not supported...");
    m_type = Type::CC;

    if(m_nAllocatedPlanSize == 0)
    {
        m_aTrajectoryAC.resize(nFrames);
        m_aTotalAC.resize(nFrames);

        m_nAllocatedPlanSize = nFrames;                
    }

    if(nFrames > m_nAllocatedPlanSize)
    {
        stringstream msg;
        msg << "Correlation: Attempting to do FFT length "
            << nFrames 
            << ", but the maximum allowed length is " 
            << m_nAllocatedPlanSize;
        THROW(out_of_range, msg.str());
    }

    m_pCorrelator->correlate(pBufferA, pBufferB, nFrames);
    for(size_t i = 0; i < nFrames; ++i)
        m_aTrajectoryAC[i] += m_pCorrelator->at(i);

    // if requested, also give the data to the optional array returning the immediate AC
    if(paThisOutput)
    {
        paThisOutput->resize(nFrames);
        for(size_t i = 0; i < nFrames; ++i)
            paThisOutput->at(i) = m_pCorrelator->at(i);
    }

    ++m_nTrajectoryCommits;
}


void Correlation::finishTrajectory(vector<double>* pTrajectoryACUnbiased)
{ 
    // copy the obtained trajectory AC and norm to the total array 
    for(size_t i = 0; i < m_nAllocatedPlanSize; ++i)
        m_aTotalAC[i] += m_aTrajectoryAC[i];

    // if requested, also store the trajectory AC (properly normed) in the given array
    if(pTrajectoryACUnbiased)
    {
        pTrajectoryACUnbiased->resize(m_nAllocatedPlanSize);
        const double dNorm = static_cast<double>(m_nTrajectoryCommits);
        for(size_t i = 0; i < m_nAllocatedPlanSize; ++i)
            pTrajectoryACUnbiased->at(i) = m_aTrajectoryAC[i] / dNorm;
    }

    // reset the trajectory arrays for the next trajectory
    memset(m_aTrajectoryAC.data(), 0, sizeof(double) * m_nAllocatedPlanSize);

    m_nTotalCommits += m_nTrajectoryCommits;
    m_nTrajectoryCommits = 0;
}

void Correlation::getCorrelation(vector<double>& aAC)
{
    // store the total AC into the given array
    aAC.resize(m_nAllocatedPlanSize);
    const double dNorm = static_cast<double>(m_nTotalCommits);
    for(size_t i = 0; i < m_nAllocatedPlanSize; ++i)
        aAC[i] = m_aTotalAC[i] / dNorm;
}

double Correlation::at(const size_t i) const 
{
    if(i >= m_aTotalAC.size())
        THROW(out_of_range, "Index ", i, " out of range of correlation function");

    const double dNorm = static_cast<double>(m_nTotalCommits);
    return m_aTotalAC[i] / dNorm;
    //return m_aTotalAC[i] / m_aTotalNorm[i];
}






///////////////////////////////////////////////////////////////////////////////////////////////////

namespace correlator
{

    Correlator::~Correlator() 
    {
    }

    CorrelatorPtr Correlator::Create(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection)
    {
        string sCorrelator = "fft";

        // find instruction in the current section
        if(pConfig->isKeyPresent(pSection, "tcf_driver"))
            sCorrelator = pConfig->get(pSection, "tcf_driver");
        else 
        {
            // look for the global default section
            if(pConfig->isSectionPresent("global"))
            {
                auto global = pConfig->getSections("global");
                if(pConfig->isKeyPresent(global.first, "tcf_driver"))
                {
                    sCorrelator = pConfig->get(global.first, "tcf_driver");
                    // update the section pointer, now pointing to the global directive
                    pSection = global.first;
                }
            }
        }

        // now create a suitable correlator
        if (sCorrelator == "fft")
            return make_shared<CorrelatorFFT>(pConfig, pSection);

        if(sCorrelator == "nemd_forward")
            return make_shared<CorrelatorNEMDForward>(pConfig, pSection);

        if(sCorrelator == "forward")            
            return make_shared<CorrelatorForward>(pConfig, pSection);

        THROW(runtime_error, "Correlator::Create: Correlator with name \"", sCorrelator, "\" not found!");
    }

    void Correlator::correlate_and_add(const double* paIn, size_t N, double* paOut)
    {
        correlate(paIn, N);
        for(size_t i = 0; i < N; ++i)
            paOut[i] += m_aWorkData[i];
    }

    void Correlator::correlate_and_add(const double* paInA, const double* paInB, size_t N, double* paOut)
    {
        correlate(paInA, paInB, N);
        for(size_t i = 0; i < N; ++i)
            paOut[i] += m_aWorkData[i];
    }

    void Correlator::check_mean(double* pData, size_t N)
    {
        double sum = 0.0;
        for(size_t i = 0; i < N; ++i)
            sum += pData[i];

        const double mean = sum / static_cast<double>(N);

        if(abs(mean) > 0.1 && !m_bMeanWarningPrinted)
        {
            cerr << "WARNING: Mean of time series for auto correlation is larger than 0.1 (" << abs(mean) << "); "
                 << "correcting for it, but might conflict physical constraints\n";
            m_bMeanWarningPrinted = true;
        }

        for(size_t i = 0; i < N; ++i)
            pData[i] -= mean;
    }

    void Correlator::check_mean(complex<double>* pData, size_t N)
    {
        double sumR = 0.0, sumI = 0.0;
        for(size_t i = 0; i < N; ++i)
        {
            sumR += pData[i].real();
            sumI += pData[i].imag();
        }

        const double meanR = sumR / static_cast<double>(N);
        const double meanI = sumI / static_cast<double>(N);
        const complex<double> mean(meanR, meanI);

        if((abs(meanR) > 0.1 || abs(meanI) > 0.1) && !m_bMeanWarningPrinted)
        {
            cerr << "WARNING: Mean of time series for cross correlation is larger than 0.1 (" << abs(meanR) << "+" << abs(meanI) << "j); "
                 << "correcting for it, but might conflict physical constraints\n";
            m_bMeanWarningPrinted = true;
        }

        for(size_t i = 0; i < N; ++i)
            pData[i] -= mean;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    CorrelatorFFT::CorrelatorFFT()
    {
    }

    CorrelatorFFT::CorrelatorFFT(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection)
    {
    }

    CorrelatorFFT::~CorrelatorFFT()
    {
        #pragma omp critical
        {
            if(m_pForwardPlan)
                fftw_destroy_plan(reinterpret_cast<fftw_plan>(m_pForwardPlan));
            if(m_pBackwardPlan)
                fftw_destroy_plan(reinterpret_cast<fftw_plan>(m_pBackwardPlan));
        }
    }

    void CorrelatorFFT::correlate(const double* pData, size_t nFrames)
    {
        // lazy initialization
        if(!m_pForwardPlan)
        {
            // allocate all buffers
            m_aWorkComplex.resize(2 * nFrames);
            m_aWorkData.resize(2 * nFrames);

            // plan creation is NEVER thread safe, even if fftw threads are initialized
            #pragma omp critical 
            {
                m_pForwardPlan = fftw_plan_dft_r2c_1d(2 * nFrames, m_aWorkData.data(), reinterpret_cast<fftw_complex*>(m_aWorkComplex.data()), FFTW_ESTIMATE);
                m_pBackwardPlan = fftw_plan_dft_c2r_1d(2 * nFrames, reinterpret_cast<fftw_complex*>(m_aWorkComplex.data()), m_aWorkData.data(), FFTW_ESTIMATE);
            }
        }

        if(!m_bAuto)
            THROW(runtime_error, "CorrelatorFFT created for cross correlations cannot be reused for auto correlations!");

        const size_t nMaxSize = m_aWorkData.size() / 2;

        if(nFrames > nMaxSize)
        {
            stringstream msg;
            msg << "Correlation: Attempting to do FFT length "
                << nFrames 
                << ", but the maximum allowed length is " 
                << nMaxSize;
            THROW(out_of_range, msg.str());
        }

        // now do the calculation!
        memset(m_aWorkData.data(), 0, sizeof(double) * 2 * nMaxSize);
        memcpy(m_aWorkData.data(), pData, sizeof(double) * nFrames);

        // check (enforce) the mean of the data
        check_mean(m_aWorkData.data(), nFrames);

        // FT for the time series
        fftw_execute(reinterpret_cast<fftw_plan>(m_pForwardPlan));

        // do the convolution
        for(size_t i = 0; i < 2 * nMaxSize; ++i)
            m_aWorkComplex[i] *= conj(m_aWorkComplex[i]);

        // back transform
        fftw_execute(reinterpret_cast<fftw_plan>(m_pBackwardPlan));
        
        // trivial normalization due to fftw
        // The norm depends on the number of bins which have been used in the FT, i.e. including zero-padding!
        // (for the CF only the number of bins are relevant which have been plugged in)
        // also apply unbiased normalization
        double dNorm = 1.0 / (static_cast<double>(2 * nMaxSize));
        for(size_t i = 0; i < nFrames; ++i)
            m_aWorkData[i] *= dNorm / (nFrames - i);
            // m_aWorkData[i] *= dNorm;
    }

    void CorrelatorFFT::correlate(const double* paInA, const double* paInB, size_t N)
    {
        // lazy initialization
        if(!m_pForwardPlan)
        {
            // allocate all buffers
            m_aWorkComplex.resize(2 * N);
            m_aWorkData.resize(2 * N);

            // plan creation is NEVER thread safe, even if fftw threads are initialized
            #pragma omp critical 
            {
                m_pForwardPlan = fftw_plan_dft_1d(2 * N, 
                    reinterpret_cast<fftw_complex*>(m_aWorkComplex.data()),
                    reinterpret_cast<fftw_complex*>(m_aWorkComplex.data()),
                    FFTW_FORWARD, FFTW_ESTIMATE);

                m_pBackwardPlan = fftw_plan_dft_c2r_1d(2 * N, 
                        reinterpret_cast<fftw_complex*>(m_aWorkComplex.data()), 
                        m_aWorkData.data(), 
                        FFTW_ESTIMATE);
            }

            // this is a cross correlation
            m_bAuto = false;
        }

        if(m_bAuto)
            THROW(runtime_error, "CorrelatorFFT created for auto correlations cannot be reused for cross correlations!");

        const size_t BufSize = m_aWorkData.size();
        const size_t nMaxSize = m_aWorkData.size() / 2;

        if(N > nMaxSize)
        {
            stringstream msg;
            msg << "Correlation: Attempting to do FFT length "
                << N 
                << ", but the maximum allowed length is " 
                << nMaxSize;
            THROW(out_of_range, msg.str());
        }

        memset(m_aWorkComplex.data() + N, 0, sizeof(complex<double>) * N);
        for(size_t i = 0; i < N; ++i)
            m_aWorkComplex[i] = complex<double>(paInA[i], paInB[i]);

        // check (enforce) the mean of the data
        check_mean(m_aWorkComplex.data(), N);

        // forward
        fftw_execute(reinterpret_cast<fftw_plan>(m_pForwardPlan));

        // convolution
        m_aWorkComplex[0] = complex<double>(real(m_aWorkComplex[0]) * imag(m_aWorkComplex[0]), 0.0);
        for(size_t i = 1; i < N; ++i)
            m_aWorkComplex[i] = 0.5 * imag(m_aWorkComplex[i] * m_aWorkComplex[BufSize - i]);
        m_aWorkComplex[N] = complex<double>(real(m_aWorkComplex[N]) * imag(m_aWorkComplex[N]), 0.0);

        // backward
        fftw_execute(reinterpret_cast<fftw_plan>(m_pBackwardPlan));
        
        // apply norm
        double dNorm = 1.0 / static_cast<double>(BufSize);
        for(size_t i = 0; i < N; ++i)
            m_aWorkData[i] *= dNorm / (N - i);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    CorrelatorForward::CorrelatorForward()
    {
    }

    CorrelatorForward::CorrelatorForward(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection)
    {
    }

    void CorrelatorForward::correlate(const double* pData, size_t N)
    {
        // doesn't actually care about the size, just resize Work array that it can hold the data
        m_aWorkData.resize(N);

        for(size_t lag = 0; lag < N; ++lag)
        {
            double sum = 0.0;
            const size_t count = N - lag;

            for(size_t i = 0; i < count; ++i)
                sum += pData[i] * pData[i + lag];

            m_aWorkData[lag] = sum / static_cast<double>(count);
        }
    }

    void CorrelatorForward::correlate(const double* paInA, const double* paInB, size_t N)
    {
        // doesn't actually care about the size, just resize Work array that it can hold the data
        m_aWorkData.resize(N);

        for(size_t lag = 0; lag < N; ++lag)
        {
            double sum = 0.0;
            const size_t count = N - lag;

            for(size_t i = 0; i < count; ++i)
                sum += paInA[i] * paInB[i + lag];

            m_aWorkData[lag] = sum / static_cast<double>(count);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    CorrelatorNEMDForward::CorrelatorNEMDForward(size_t nT0AvgLength)
        : m_nT0AvgLength(nT0AvgLength)
    {
    }

    CorrelatorNEMDForward::CorrelatorNEMDForward(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection)
        : m_nT0AvgLength(pConfig->as<size_t>(pSection, "tcf_driver_nemd_forward_t0_len", 1))
    {
    }

    void CorrelatorNEMDForward::correlate(const double* pData, size_t N)
    {
        // doesn't actually care about the size, just resize Work array that it can hold the data
        m_aWorkData.resize(N);

        // here NO (limited) averaging over initial times
        for(size_t lag = 0; lag < N; ++lag)
        {
            double sum = 0.0;
            const size_t count = min(N - lag, m_nT0AvgLength);

            for(size_t i = 0; i < count; ++i)
                sum += pData[i] * pData[i + lag];

            m_aWorkData[lag] = sum / static_cast<double>(count);
        }
    }

    void CorrelatorNEMDForward::correlate(const double* paInA, const double* paInB, size_t N)
    {
        // doesn't actually care about the size, just resize Work array that it can hold the data
        m_aWorkData.resize(N);

        // here NO (limited) averaging over initial times
        for(size_t lag = 0; lag < N; ++lag)
        {
            double sum = 0.0;
            const size_t count = min(N - lag, m_nT0AvgLength);

            for(size_t i = 0; i < count; ++i)
                sum += paInA[i] * paInB[i + lag];

            m_aWorkData[lag] = sum / static_cast<double>(count);
        }
    }

}
