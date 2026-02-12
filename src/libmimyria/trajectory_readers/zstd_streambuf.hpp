#ifndef __ZSTD_STREAMBUF_HPP__
#define __ZSTD_STREAMBUF_HPP__

#include "../pch.hpp"

#if HAVE_ZSTD 

#include <zstd.h>

class ZStdStreamBuf : public std::streambuf
{
public:
    explicit ZStdStreamBuf(std::istream* pIn)
        : m_pIn(pIn)
        , dctx(ZSTD_createDCtx())
        , inBuffer(IN_CHUNK_SIZE)
        , outBuffer(OUT_CHUNK_SIZE)  
    {
        input.src = inBuffer.data();
        input.pos = input.size = 0;

        output.dst = outBuffer.data();
        output.size = outBuffer.size();
        output.pos = 0;

        setg(outBuffer.data(), outBuffer.data(), outBuffer.data());
    }

    virtual ~ZStdStreamBuf()
    {
        ZSTD_freeDCtx(dctx);
    }

    virtual int underflow() override final 
    {
        if(gptr() < egptr())
            return traits_type::to_int_type(*gptr());

        // loop until output buffer contains results
        output.pos = 0;
        while(output.pos == 0)
        {
            // only read new data, if input buffer fully read
            if(input.pos == input.size)
            {
                m_pIn->read(inBuffer.data(), inBuffer.size());
                input.size = m_pIn->gcount();
                input.pos = 0;

                if(input.size == 0) 
                    return traits_type::eof();
            }

            size_t ret = ZSTD_decompressStream(dctx, &output, &input);

            if(ZSTD_isError(ret))
                throw std::runtime_error("ZSTD Decompression failed: " + std::string(ZSTD_getErrorName(ret)));

            setg(outBuffer.data(), outBuffer.data(), outBuffer.data() + output.pos);
        }

        return traits_type::to_int_type(*gptr());
    }

private:
    std::istream* m_pIn;

    static constexpr size_t IN_CHUNK_SIZE = 256 * 1024;
    static constexpr size_t OUT_CHUNK_SIZE = 512 * 1024;
    //static constexpr size_t IN_CHUNK_SIZE = 16384;
    //static constexpr size_t OUT_CHUNK_SIZE = 65536;

    ZSTD_DCtx* dctx;
    std::vector<char> inBuffer; 
    std::vector<char> outBuffer;
    ZSTD_inBuffer input;
    ZSTD_outBuffer output;
};

#endif // HAVE_ZSTD

#endif // once
