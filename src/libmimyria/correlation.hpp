#ifndef __CORRELATION_HPP__
#define __CORRELATION_HPP__

#include "pch.hpp"
#include "configfile.hpp"

namespace correlator
{
    // driver which actually does the correlation
    class Correlator
    {
    public:
        virtual ~Correlator() = 0;

        // creates a suitable correlator based on the configuration file
        static std::shared_ptr<Correlator> Create(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection);

        // clones this correlator and returns a new copy
        virtual std::shared_ptr<Correlator> clone() const = 0;

        // access to the internal buffer containing the computed correlation function
        inline double at(size_t i) const { return m_aWorkData.at(i); }

        // Do an AUTO correlation
        virtual void correlate(const double* pData, size_t N) = 0; 
        // Do a CROSS correlation
        virtual void correlate(const double* pDataA, const double* pDataB, size_t N) = 0;

        // Do an AUTO correlation
        virtual void correlate_and_add(const double* pData, size_t N, double* paOut);
        // Do a CROSS correlation
        virtual void correlate_and_add(const double* pDataA, const double* pDataB, size_t N, double* paOut);

    protected:
        void check_mean(double* pData, size_t N);
        void check_mean(std::complex<double>* pData, size_t N);

        std::vector<double> m_aWorkData;
        bool m_bMeanWarningPrinted{false};
    };
    typedef std::shared_ptr<Correlator> CorrelatorPtr;

    // Correlation via convolution and FFT, scales N log N
    class CorrelatorFFT : public Correlator 
    {
    public:
        CorrelatorFFT();
        CorrelatorFFT(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection);
        virtual ~CorrelatorFFT();
        virtual std::shared_ptr<Correlator> clone() const override { return std::make_shared<CorrelatorFFT>(); }
        virtual void correlate(const double* pData, size_t nFrames) override;
        virtual void correlate(const double* pDataA, const double* pDataB, size_t N) override;

    private:
        // working buffers of FFTW
        std::vector<std::complex<double>> m_aWorkComplex;
        // FFTW plans
        void *m_pForwardPlan{nullptr}, *m_pBackwardPlan{nullptr};
        // flag keeping track on what type this correlator is (cross/auto) since in fftw they are not interchangeable
        bool m_bAuto{true};
    };

    // default direct correlation function, should give the same as FFT, but scales N^2
    class CorrelatorForward : public Correlator 
    {
    public:
        CorrelatorForward();
        CorrelatorForward(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection);

        virtual std::shared_ptr<Correlator> clone() const override { return std::make_shared<CorrelatorForward>(); }
        virtual void correlate(const double* pData, size_t nFrames) override;
        virtual void correlate(const double* pDataA, const double* pDataB, size_t N) override;
    };

    // forward two point correlation function
    class CorrelatorNEMDForward : public Correlator 
    {
    public:
        // The parameter decides over how many lags the initial time should be averaged over
        // For a forward TCF, 1 is the proper choice, however, proper sampling necessitates very many trajectories...
        CorrelatorNEMDForward(size_t nT0AvgLength = 1);
        CorrelatorNEMDForward(ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection);
        virtual std::shared_ptr<Correlator> clone() const override { return std::make_shared<CorrelatorNEMDForward>(m_nT0AvgLength); }
        virtual void correlate(const double* pData, size_t nFrames) override;
        virtual void correlate(const double* pDataA, const double* pDataB, size_t N) override;

    private:
        const size_t m_nT0AvgLength{1};
    };
}

///////////////////////////////////////////////////////////////////////////////////////////////////
class Correlation 
{
public:
    //////////////////////////////////////////////

    // driver which actually does the correlation
    typedef correlator::CorrelatorFFT CorrelatorFFT;
    typedef correlator::CorrelatorForward CorrelatorForward;
    typedef correlator::CorrelatorNEMDForward CorrelatorNEMDForward;

    //////////////////////////////////////////////
    Correlation(std::unique_ptr<correlator::Correlator> pTCFDriver = nullptr); 
    ~Correlation();
    
    // only move, no copy
    Correlation(const Correlation& rhs) = delete;
    Correlation(Correlation&& rhs) = default;

    // returns value of ac at lag i
    double at(const size_t i) const;

    // commits raw data, computes the correlation function and stores it in the internal buffer,
    // the first call determines AC or CC, the buffer lengths, afterwards nFrames must be equal or less
    // if set, paThisOutput will return the correlation function of this commit only
    void commit(const double* paBuffer, size_t nFrames, std::vector<double>* paThisOutput = nullptr);

    // commits raw data, computes the cross correlation function and stores it in the internal buffer, 
    // the first call determines AC or CC, the buffer lengths, afterwards nFrames must be equal or less
    // if set, paThisOutput will return the correlation function of this commit only
    void commit(const double* paBufferA, const double* paBufferB, size_t nFrames, std::vector<double>* paThisOutput = nullptr);

    // closes processing of the current trajectory. 
    // as a consequence all correlation functions committed so far will be added to the total buffer
    // if set, paTrajectoryAC returns the averaged correlation function of the last trajectory
    void finishTrajectory(std::vector<double>* paTrajectoryACUnbiased = nullptr);

    // returns the total correlation function
    // output is normalized and averaged
    void getCorrelation(std::vector<double>& aAC);

    // returns the numbers of commits
    inline size_t getNumCommits() const { return m_nTotalCommits; }
    // returns the number of maximal allocated frames
    inline size_t getNumFrames() const { return m_nAllocatedPlanSize; }

    // returns value of AC at lag i
    inline double operator[](const size_t i) const { return at(i); }

private:
    // the AC correlation buffers
    std::vector<double> m_aTrajectoryAC, m_aTotalAC;
    // normalization factor for the number of commits
    size_t m_nTrajectoryCommits{0}, m_nTotalCommits{0};

    size_t m_nAllocatedPlanSize{0};

    // the correlator used
    std::unique_ptr<correlator::Correlator> m_pCorrelator;

    enum Type
    {
        // not yet defined
        Undefined, 
        // used for auto correlations
        AC,
        // used for cross correlations 
        CC 
    }
    m_type;
};

typedef std::unique_ptr<Correlation> CorrelationPtr;
typedef CorrelationPtr AutoCorrelationPtr;
typedef CorrelationPtr CrossCorrelationPtr;

typedef Correlation AutoCorrelation;
typedef Correlation CrossCorrelation;

#endif
