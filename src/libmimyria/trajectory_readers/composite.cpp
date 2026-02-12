#include "../pch.hpp"
#include "../exception.hpp"
#include "../trajectory.hpp"
#include <regex>

using namespace std;

class CompositeTrajectory : public Trajectory 
{
public:
    CompositeTrajectory()
    { 
    }
    virtual ~CompositeTrajectory()
    { 
        close();
    }

    // close trajectory
    virtual void close() override
    {
    }

    virtual void open(std::istream* pisInput, const std::string& sName, const KVPairs& extra) override
    {
        // cannot open a file Stream!
        THROW(runtime_error, "CompositeTrajectory cannot be opened by a file stream!");
    }

    virtual void open(const std::string& line, const KVPairs& extra) override 
    {
        // this is a comma-separated string, extract each element
        sregex_iterator beg(line.begin(), line.end(), tokenRegex);
        for(auto tokenIt = beg; tokenIt != sregex_iterator(); ++tokenIt)
        {
            string fn = tokenIt->str();

            KVPairs extras;

            size_t openParen = fn.find('(');
            if(openParen != string::npos)
            {
                string kvpart = fn.substr(openParen + 1, fn.find(')'));
                sregex_iterator kvbeg (kvpart.begin(), kvpart.end(), keyValueRegex);

                for(auto kv = kvbeg; kv != sregex_iterator(); ++kv)
                    extras.insert(make_pair(kv->str(1), kv->str(2)));

                fn = fn.substr(0, openParen);
            }

            trim(fn);
            
            // create the trajectory and open 
            auto ext = fn.substr(fn.rfind('.')+1);

            // open trajectory and store
            TrajectoryPtr pTraj = registry::TrajectoryFactory::Get().Create(ext);
            pTraj->open(fn, extras);
            m_aTrajectories.push_back(pTraj);
        }

        Trajectory::open(nullptr, line);
    }

    // read next frame
    virtual FramePtr readNextFrame()  override
    {
        if(m_aTrajectories.empty())
            THROW(runtime_error, "CompositeTrajectoryReader::readNextFrame(): No trajectory opened yet!");

        // NOTE prioritizing the first trajectory!
        // ONLY overwrite if additional information is added by the following trajectories
        FramePtr pFrame; 
        for(auto pTraj : m_aTrajectories)
        {
            FramePtr pOther = pTraj->readNextFrame();

            // Return EOF if any trajectory is EOF
            if(!pOther)
                return nullptr;

            if(!pFrame)
                pFrame = pOther;
            else 
                pFrame->add(pOther);
        }

        return pFrame;
    }

    // ESTIMATE the number of frames contained in the opened stream or file. 
    // not 100 accurate, but gives a rough estimation which is surprisingly accurate for XYZ files.
    virtual size_t estimateTotalNumberOfFrames() override
    {
        // prioritize the first frame
        return m_aTrajectories.front()->estimateTotalNumberOfFrames();
    }

private: 
    static const regex tokenRegex;
    static const regex keyValueRegex;

    vector<TrajectoryPtr> m_aTrajectories;
};

const regex CompositeTrajectory::tokenRegex(R"(([^,]+))");
const regex CompositeTrajectory::keyValueRegex(R"((\w+)=([\w:.]+))");


REGISTER(registry::TrajectoryFactory, "composite", CompositeTrajectory);

