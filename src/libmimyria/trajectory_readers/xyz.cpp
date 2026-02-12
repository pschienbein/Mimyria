#include "../pch.hpp"
#include "../exception.hpp"
#include "../trajectory.hpp"
#include <chrono>
#include <regex>

using namespace std;

class XYZTrajectory : public Trajectory 
{
public:
    XYZTrajectory()
        : m_nExpectedCols(0)
    { 
    }
    virtual ~XYZTrajectory()
    { 
        close();
    }

    // close trajectory
    virtual void close() override
    {
    }

    // open a c++ stream
    virtual void open(std::istream* pisInput, const std::string& sName, const KVPairs& extra) override
    {
        // Base class initialization first
        Trajectory::open(pisInput, sName);

        // create custom comment line if necessary, in particular relevant is the Properties key if present:
        for(auto kvpair : extra)
        {
            if(kvpair.first == "Properties")
            {
                // this determines the default properties to be read
                parse_properties(kvpair.second, m_aDefaultReadInstructions);
            }
        }
    }

    // read next frame
    virtual FramePtr readNextFrame() override
    {

        auto start = chrono::high_resolution_clock::now();

        // store read data in Frame object
        FramePtr pFrame = make_shared<Frame>();

        size_t nLine = 0, nAtoms = 0;
        string line;
        KVPairs commentKV;
        ReadInstructionList aReadInstructions;
        while(getline(*m_pisInput, line))
        {
            try 
            {
                istringstream iss_line(line);
                iss_line.exceptions(ios::failbit | ios::badbit);

                if(nLine == 0)
                {
                    try {
                        iss_line >> nAtoms;
                    }
                    catch(ios_base::failure& e)
                    {
                        THROW(runtime_error, "XYZParser: Could not read number of atoms in frame; potentially a mismatch between the number of atoms announced and the actual number of atoms?");
                    }
                }
                else if(nLine == 1)
                {
                    try 
                    {
                        // comment line
                        parse_comment_line(line, commentKV);
        //                cerr << "Comment line: " << line << endl;                  
                        // are properties provided? 
                        auto pItem = commentKV.find("Properties");
                        if(pItem != commentKV.end())
                        {
        //                    cerr << "Properties provided: " << pItem->second << endl;
                            parse_properties(pItem->second, aReadInstructions);
                        }
                        else 
                        {
        //                    cerr << "Properties not provided, using default\n";
                            copy(m_aDefaultReadInstructions.begin(), m_aDefaultReadInstructions.end(), back_inserter(aReadInstructions));
                        }

                        if(aReadInstructions.empty())
                            THROW(runtime_error, "No Reading instructions given in the comment line or when opening the trajectory!");

                        // allocate buffers
                        for(auto ri : aReadInstructions)
                        {
                            switch(ri)
                            {
                            case ReadInstruction::Species:
                                pFrame->m_sSymbols.resize(nAtoms);
                                break;
                            case ReadInstruction::Position:
                                pFrame->m_vPositions.resize(nAtoms);
                                break;
                            case ReadInstruction::Velocity:
                                pFrame->m_vVelocities.resize(nAtoms);
                                break;
                            case ReadInstruction::APT:
                                pFrame->m_mAPT.resize(nAtoms);
                                break;
                            case ReadInstruction::PGT:
                                pFrame->m_mPGT.emplace(nAtoms);
                                break;
                            case ReadInstruction::AtomDipole:
                                pFrame->m_vAtomDipoles.emplace(nAtoms);
                                break;
                            }
                        }
                    }
                    catch(ios_base::failure& e)
                    {
                        THROW(runtime_error, "XYZParser: Could not parse comment line");
                    }
                }
                else 
                {
                    // "normal" atom

                    const size_t iAtom = nLine - 2;
                    if(aReadInstructions.empty())
                        THROW(runtime_error, "No Reading instructions given in the comment line or when opening the trajectory!");

                    try
                    {
                        for(auto ri : aReadInstructions)
                        {
                            switch(ri)
                            {
                            case ReadInstruction::Species:
                                iss_line >> pFrame->m_sSymbols[iAtom];
                                break;

                            case ReadInstruction::Position:
                                for(size_t k = 0; k < 3; ++k)
                                    iss_line >> pFrame->m_vPositions[iAtom][k];
                                break;

                            case ReadInstruction::Velocity:
                                for(size_t k = 0; k < 3; ++k)
                                    iss_line >> pFrame->m_vVelocities[iAtom][k];
                                break;

                            case ReadInstruction::AtomDipole:
                                for(size_t k = 0; k < 3; ++k)
                                    iss_line >> (*pFrame->m_vAtomDipoles)[iAtom][k];
                                break;

                            case ReadInstruction::APT:
                                for(size_t j = 0; j < 3; ++j)
                                {
                                    for(size_t k = 0; k < 3; ++k)
                                        iss_line >> pFrame->m_mAPT[iAtom](k,j);
                                }
                                break;

                            case ReadInstruction::PGT:
                                for(size_t j = 0; j < 3; ++j)
                                {
                                    for(size_t k = 0; k < 3; ++k)
                                    {
                                        for(size_t x = 0; x < 3; ++x)
                                            iss_line >> (*pFrame->m_mPGT)[iAtom][x](j, k);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    catch(ios_base::failure& e)
                    {
                        THROW_WITH_NESTED(runtime_error, "XYZParser: Stream parsing failed at atom ", iAtom);
                    }

                    if(iAtom + 1 == nAtoms)
                    {
                        //auto end = chrono::high_resolution_clock::now();
                        //chrono::duration<double> dt1 = end - start;
                        //cerr << "C1: " << dt1.count();

                        return pFrame;
                    }
                }
                ++nLine;
            }
            catch(exception& e)
            {
                THROW_WITH_NESTED(runtime_error, "XYZParser: Stream parsing failed, line was \"", line, "\"");
            }
         } // while

        return nullptr;
    }

    // ESTIMATE the number of frames contained in the opened stream or file. 
    // not 100 accurate, but gives a rough estimation which is surprisingly accurate for XYZ files.
    virtual size_t estimateTotalNumberOfFrames() override 
    {
    }

private: 
    string m_sCustomComment;

    enum class ReadInstruction 
    {
        Species,
        Position, 
        Velocity,
        APT,
        PGT,
        AtomDipole // ... as product of APT and velocity
    };
    typedef vector<ReadInstruction> ReadInstructionList;
    ReadInstructionList m_aDefaultReadInstructions;

    // expected number of columns based on the comment line
    size_t m_nExpectedCols;

    inline void parse_comment_line(const string& comment, KVPairs& kv)
    {
        static thread_local regex pattern(R"(([^=\s]+)=(\".*?\"|\S+))");

        smatch match;
        auto begin = comment.cbegin();

        while(regex_search(begin, comment.cend(), match, pattern))
        {
            string key = match[1];                
            string val = match[2];

            // strip quotes
            if(!val.empty() && val.front() == '"' && val.back() == '"')
                val = val.substr(1, val.length() - 2);

            kv.insert(make_pair(key, val));

            begin = match.suffix().first;
        }
    }

    inline void parse_properties(const string& sPropString, ReadInstructionList& ri)
    {
        istringstream prop(sPropString);
        string token;
        size_t n = 0;
        while(getline(prop, token, ':')) 
        {
            if(n % 3 == 0)
            {
                // translate from string to ReadInstruction
                if(token == "species")
                {
                    ri.push_back(ReadInstruction::Species);
                    m_nExpectedCols += 1;
                }
                else if(token == "pos")
                {
                    ri.push_back(ReadInstruction::Position);
                    m_nExpectedCols += 3;
                }
                else if(token == "vel")
                {
                    ri.push_back(ReadInstruction::Velocity);
                    m_nExpectedCols += 3;
                }
                else if(token == "apt")
                {
                    ri.push_back(ReadInstruction::APT);
                    m_nExpectedCols += 9;
                }
                else if(token == "pgt")
                {
                    ri.push_back(ReadInstruction::PGT);
                    m_nExpectedCols += 27;
                }
                else if(token == "dip")
                {
                    ri.push_back(ReadInstruction::AtomDipole);
                    m_nExpectedCols += 3;
                }
                else 
                    cerr << "WARNING: Unknown format specifier \"" << token << "\"";
            }
            ++n;
        }
    }
};


REGISTER(registry::TrajectoryFactory, "xyz", XYZTrajectory);
