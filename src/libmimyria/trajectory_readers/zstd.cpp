#include "../pch.hpp"
#include "zstd_streambuf.hpp"
#include "../exception.hpp"
#include "../trajectory.hpp"

#if HAVE_ZSTD

using namespace std;

// This parser is just used to decompress a given Zstd file and hands the results to another nested parser which is used to parse the data
class ZStdTrajectoryInputParser : public Trajectory
{
public:
    // closes this parser
    virtual void close() override
    {
        if(m_pNestedParser)
            m_pNestedParser->close();
    }

    // setup this parser
    virtual void open(const std::string& sFilename, const KVPairs& cextra) override
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
        m_pFileDecoder.reset(new ZStdStreamBuf(m_pisInput));
        m_pisDecompressed.reset(new istream(m_pFileDecoder.get()));

        // finally setup the nested parser
        m_pNestedParser->open(m_pisDecompressed.get(), getName() + "->Nested", extra);
    }

    // reads the next frame from the input
    virtual FramePtr readNextFrame() override
    {
        return m_pNestedParser->readNextFrame();
    }

private:
    // Nested parser which is used to parse the decompressed data
    TrajectoryPtr m_pNestedParser;
    // File stream containing the decompressed data
    std::unique_ptr<std::istream> m_pisDecompressed;
    // Actual decoder
    std::unique_ptr<std::streambuf> m_pFileDecoder;
};

// Register this class!
REGISTER(registry::TrajectoryFactory, "zst", ZStdTrajectoryInputParser);
REGISTER(registry::TrajectoryFactory, "zstd", ZStdTrajectoryInputParser);

#endif // HAVE_ZSTD
