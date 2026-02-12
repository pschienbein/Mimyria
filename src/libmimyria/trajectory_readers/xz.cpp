#include "../pch.hpp"
#include "lzma_streambuf.hpp"
#include "../exception.hpp"
#include "../trajectory.hpp"

using namespace std;

#if HAVE_LZMA

class LZMAReader : public Trajectory
{
public:
    LZMAReader() 
        : Trajectory()
    {
    }

    virtual ~LZMAReader()
    {
        close();
    }

    // closes this parser
    void close() override
    {
        if(m_pNestedParser)
            m_pNestedParser->close();
    }

    // open file
    void open(const std::string& sFilename, const KVPairs& cextra)  override
    {
        m_ifsInFile.open(sFilename.c_str(), ios::in | ios::binary);
        if(m_ifsInFile.bad() || m_ifsInFile.fail())
            THROW(runtime_error, "Specified input file \"", sFilename, "\" could not be opened!");

        // Remove the file extension
        string sFnWithoutExt = sFilename.substr(0, sFilename.find_last_of('.'));

        // the extra arguments passed should contain the extension of the nested trajectory; 
        // if not, it is determined from the file name
        KVPairs extra = cextra;
        if(extra.find("ext") == extra.end())
        {
            // the next extension determines which file to load
            auto sExt = sFnWithoutExt.substr(sFnWithoutExt.find_last_of('.') + 1);
            extra.insert(make_pair("ext", sExt));
        }

        // forward
        open(&m_ifsInFile, sFilename, extra);
    }

    void open(std::istream* pisInput, const std::string& sName, const KVPairs& extra) override
    {
        // Base class
        Trajectory::open(pisInput, sName, extra);

        // get ext from kv pairs
        auto pItem = extra.find("ext");
        if(pItem == extra.end())
            THROW(runtime_error, "LZMAReader::open: An extension needs to be set via extra[\"ext\"] in order to determine which parser is to be used after decompressing");
        auto fileext = pItem->second;

        // load the nested parser 
        try 
        {	
            m_pNestedParser = registry::TrajectoryFactory::Get().Create(fileext);
        }
        catch(runtime_error&)
        {
            THROW_WITH_NESTED(runtime_error, "Could not find a suitable input parser for file extension \"", fileext, "\"");
        }

        // Create the decoder and the associated stream for reading
        m_pFileDecoder.reset(new LZMAStreamBuf(m_pisInput));
        m_pisDecompressed.reset(new istream(m_pFileDecoder.get()));

        // finally setup the nested parser
        m_pNestedParser->open(m_pisDecompressed.get(), getName() + "->Nested", extra);
    }

    // reads the next frame from the input
    FramePtr readNextFrame() override
    {
        // pass task to the nested interpreter
        return m_pNestedParser->readNextFrame();
    }

private:
    TrajectoryPtr m_pNestedParser;
    unique_ptr<LZMAStreamBuf> m_pFileDecoder;
    unique_ptr<istream> m_pisDecompressed;
};

// Register this class!
REGISTER(registry::TrajectoryFactory, "xz", LZMAReader);

#endif
