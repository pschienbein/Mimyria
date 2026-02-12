#include "pch.hpp"

#include <chrono>
#include <execinfo.h>
#include <unistd.h>
#include <csignal>

#include "exception.hpp"
#include "task.hpp"
#include "trajectory.hpp" 
#include "thread_queue.hpp" 
#include "configfile.hpp"
#include "plugin.hpp"
#include "library.hpp"

#include "box_loader.hpp"

using namespace std;

void ReadingThreadFunc(TrajectoryPtr pTrajectory, shared_ptr<ThreadQueue<FramePtr>> pStorage)
{
    try 
    {
        bool bExit = false;
        FramePtr pFrame = nullptr;
        while((pFrame = pTrajectory->readNextFrame()) != nullptr && !bExit)
            bExit = !pStorage->push(pFrame);
    }
    catch(exception& e)
    {
        print_nested_exception(cerr, e);
        THROW(runtime_error, "Parsing of Trajectory ", pTrajectory->getName(), " failed\n");
    }
    
    pStorage->exit();
}

void handleSegFault(int signal)
{
    cerr << "Segmentation Fault caught (signal: " << signal << "), Thread-ID: " << this_thread::get_id() << endl;

    //exit(signal);

    const int maxFrames = 100; 
    void* frames[maxFrames];
    int nFrames = backtrace(frames, maxFrames);

    cerr << "Callstack:\n";
    char** symbols = backtrace_symbols(frames, nFrames);
    if(symbols)
    {
        for(int i = 0; i < nFrames; ++i)
            cerr << symbols[i] << endl;
        free(symbols);
    }
    else
        cerr << "Failed to retrieve symbols for backtrace\n";

    exit(signal);
}


template<class T>
void overwrite_dataset(library::Dataset<T>& ds, ConfigFilePtr pConfig, ConfigFile::SectionPtr pSection)
{
    const auto keys = pConfig->keys(pSection);
    for(const auto& key : keys)
        ds.overwrite(key, pConfig->as<T>(pSection, key));
}


int main(int argc, char** argv)
{ 
    struct sigaction action; 
    action.sa_sigaction = [](int sig, siginfo_t*, void*){
        handleSegFault(sig);
    };
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);

    signal(SIGSEGV, handleSegFault);
    signal(SIGABRT, handleSegFault);
    
    struct NotInitializedTrajectory
    {
        TrajectoryPtr pTrajectory;
        string sInitCmd;
    };
    queue<NotInitializedTrajectory> aTrajectories;

    try 
    {
        cxxopts::Options options("SpectraNova", "Description");

        options.add_options()
            ("help", "Print this message and exit")
            ("config", "ini file containing detailed configuration information for the specific task", cxxopts::value<string>())
            ("files", "List of input trajectory files to process", cxxopts::value<vector<string>>())
            ("filelist", "Text file containing a list of files to process, allows to open many individual files as one trajectory, useful if positions, velocities, apts, ... are stored in different files", cxxopts::value<string>())
            ;

        options.parse_positional({"config", "files"});

        cxxopts::ParseResult parsed_arguments = options.parse(argc, argv);

        // Print help string and exit if required
        if(parsed_arguments.count("help"))
        {
            cerr << options.help() << endl;
            return 0;
        }

        // intialize all requested files
        // try to open all of them to find a faulty/missing trajectory BEFORE starting processing them
        // NOTE: close them again, to not have thousands of files open at once
        if(parsed_arguments.count("files") > 0)
        {
            auto files = parsed_arguments["files"].as<vector<string>>();
            for(const auto& file : files)
            {
                auto ext = file.substr(file.rfind('.')+1);

                auto pTraj = registry::TrajectoryFactory::Get().Create(ext);
                // NOTE: test if the file is existing
                aTrajectories.emplace(NotInitializedTrajectory{pTraj, file});
            }
        }
        else 
        {   
            // Test if filelist is given
            if(parsed_arguments.count("filelist") > 0)
            {
                string fn = parsed_arguments["filelist"].as<string>();
                ifstream ifs(fn, ios::in);
                if(!ifs.good())
                    THROW(runtime_error, "Could not open file \"", fn, "\"");

                string line, token; 
                while(getline(ifs, line))
                {
                    auto pTraj = registry::TrajectoryFactory::Get().Create("composite");
                    // NOTE: test if the file is existing
                    aTrajectories.emplace(NotInitializedTrajectory{pTraj, line});
                }
            }
            else 
            {
                cerr << "No input files and no filelist provided, exiting\n";
                return 1;
            }
        }

        if(aTrajectories.empty())
        {
            cerr << "No input files provided, or filelist is empty, exiting\n";
            return 1;
        }

        // load configuration file
        ConfigFilePtr pConfig;
        if(parsed_arguments.count("config"))
        {
            pConfig = make_shared<ConfigFile>();
            pConfig->open(parsed_arguments["config"].as<string>());
        }
        else
            THROW(runtime_error, "config file needs to be provided");

        // Load plugins (if any)
        Plugins::Load(pConfig);

        // override / add masses (if any)
        if(pConfig->isSectionPresent("mass"))
            overwrite_dataset(library::Masses, pConfig, pConfig->getSection("mass"));
        if(pConfig->isSectionPresent("charge"))
            overwrite_dataset(library::Charges, pConfig, pConfig->getSection("charge"));

        // create tasks from config file
        vector<TaskPtr> aTasks;

        // magic sections to be ignored for TASK creation
        set<string> aSection2Ignore;
        aSection2Ignore.insert("global");
        aSection2Ignore.insert("plugin");
        aSection2Ignore.insert("cell");
        aSection2Ignore.insert("mass");
        aSection2Ignore.insert("charge");

        auto sections = pConfig->getSections();
        for(auto pSection = sections.first; pSection != sections.second; ++pSection)
        {
            const auto& sSectionName = pSection->first; 

            if(auto search = aSection2Ignore.find(sSectionName); search != aSection2Ignore.end())
                continue;
            
            // try to create a task:
            TaskPtr pTask = registry::TaskFactory::Get().Create(sSectionName);
            // setup
            pTask->setup(pConfig, pSection);
            // add to array 
            aTasks.push_back(pTask);
        }

        if(aTasks.empty())
            THROW(runtime_error, "at least one task needs to be defined in the config file");

        // Create Box Loader
        BoxLoader box_loader;
        box_loader.setup(pConfig);

        // loop over all files
        size_t nFrames = 0;
        size_t nQueueSizeMean = 0;

        // flag to be moved to the control file
        bool bPreloadNextTrajectory = true;
        // number to be moved to the control file
        size_t nFrameBufferSize = 30000;

        auto pnit = &(aTrajectories.front());
        pnit->pTrajectory->open(pnit->sInitCmd);

        auto pFrameStorage = make_shared<ThreadQueue<FramePtr>>(nFrameBufferSize);
        auto pReadThread = make_shared<thread>(ReadingThreadFunc, pnit->pTrajectory, pFrameStorage);

        while(!aTrajectories.empty())
        {
            cerr << "Proceeding to " << pnit->pTrajectory->getName() << endl;

            try 
            {
                box_loader.nextTrajectory();

                // notify task
                for(auto pTask : aTasks)
                    pTask->onTrajectoryOpened();

                // timing
                auto t1 = chrono::high_resolution_clock::now();

                // loop over all frames:
                while(true)
                {
                    // for reporting purposes
                    nQueueSizeMean += pFrameStorage->size();

                    auto ppFrame = pFrameStorage->pop();
                    if(!ppFrame)
                        break;
                    BoxPtr pBox = box_loader.get(*ppFrame);
                    (*ppFrame)->m_pBox = pBox;

                    for(auto pTask : aTasks)
                        pTask->onFrame(*ppFrame);

                    ++nFrames;
                    if(nFrames % 1000 == 0)
                        cerr << "Processed " << nFrames << " Frames\r";
                }
                cerr << "Processed " << nFrames << " Frames\n";

                // timing
                auto t2 = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
                cout << "Trajectory IO took " << static_cast<double>(duration.count()) / 1000.0 << " s, Mean queue size: " << static_cast<double>(nQueueSizeMean) / nFrames << endl;

                // if no preloading, do the postprocessing now,
                // create the next reading thread LATER
                if(!bPreloadNextTrajectory)
                {
                    // notify task
                    for(auto pTask : aTasks)
                        pTask->onTrajectoryFinished();
                }

                // wait for the reading thread to finish
                pReadThread->join();

                // open the next trajectory for processing, if available
                pnit = nullptr;
                aTrajectories.pop();

                if(!aTrajectories.empty())
                {
                    pnit = &aTrajectories.front();
                    pnit->pTrajectory->open(pnit->sInitCmd);

                    pFrameStorage = make_shared<ThreadQueue<FramePtr>>(nFrameBufferSize);
                    pReadThread = make_shared<thread>(ReadingThreadFunc, pnit->pTrajectory, pFrameStorage);
                }

                // if preloading, the new reading thread has been opened, 
                // do the postprocessing now
                if(bPreloadNextTrajectory)
                {
                    // notify task
                    for(auto pTask : aTasks)
                        pTask->onTrajectoryFinished();
                }

                // timing
                auto t3 = chrono::high_resolution_clock::now();
                duration = chrono::duration_cast<chrono::milliseconds>(t3 - t2);
                cout << "Trajectory postprocessing took " << static_cast<double>(duration.count()) / 1000.0 << " s\n";
            }
            catch(const exception& e)
            {
                cerr << "Pending exception, waiting for reading thread to finish ...\n";
                pFrameStorage->exit();
                pReadThread->join();

                // try to finish stuff
                try
                {
                    // notify task
                    for(auto pTask : aTasks)
                        pTask->onFinished();
                }
                catch(exception&)
                {
                    // nothing here, just ensure that the main exception gets thrown again
                }

                THROW_WITH_NESTED(runtime_error, "Task Processing failed");
            }
        }

        // notify task
        for(auto pTask : aTasks)
            pTask->onFinished();

        // Print finalizing message
        cout << "Mean frame queue length: " << static_cast<double>(nQueueSizeMean) / static_cast<double>(nFrames) << endl;
    }
    catch(exception& e)
    {
        print_nested_exception(cerr, e);
    }

    return 0;
}
