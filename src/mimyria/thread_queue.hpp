#ifndef __THREAD_QUEUE_HPP__
#define __THREAD_QUEUE_HPP__

#include "pch.hpp"

template<class TData>
class ThreadQueue 
{
public:
    inline ThreadQueue(size_t nMaxQueueSize)
        : m_nMaxQueueSize(nMaxQueueSize)
        , m_bExit(false)
    {
    }

    inline ~ThreadQueue() 
    {
        exit();
    }

    inline void exit()
    {
        std::unique_lock<std::mutex> lock(m_accessMutex);
        m_bExit = true;
        m_cvDataAdded.notify_all();
        m_cvDataRemoved.notify_all();
    }

    inline bool push(const TData& data)
    {
        std::unique_lock<std::mutex> lock(m_accessMutex);

        if(m_queue.size() >= m_nMaxQueueSize)
        {
            // wait until either enough items have been removed from the queue, or the thread should exit
            m_cvDataRemoved.wait(lock, [this] { return m_queue.size() < m_nMaxQueueSize || m_bExit; });
        }

        if(m_bExit)
            return false;

        m_queue.push(data);
        m_cvDataAdded.notify_one();
        return true;
    }

    inline std::optional<TData> pop()
    {
        std::unique_lock<std::mutex> lock(m_accessMutex);
        if(m_queue.empty() && m_bExit)
            return std::nullopt;

        if(m_queue.empty())
            m_cvDataAdded.wait(lock, [this]() { return !m_queue.empty() || m_bExit; });

        if(m_queue.empty())
            return std::nullopt;

        auto item = std::move(m_queue.front());
        m_queue.pop();

        // notify the IO trhead to provide more data
        m_cvDataRemoved.notify_one();

        return item;
    }

    inline size_t size() const 
    {
        std::unique_lock<std::mutex> lock(m_accessMutex);
        return m_queue.size();
    }

private:
    std::queue<TData> m_queue;
    mutable std::mutex m_accessMutex;
    std::condition_variable m_cvDataAdded;
    std::condition_variable m_cvDataRemoved;
    const size_t m_nMaxQueueSize;
    bool m_bExit;
};

#endif 
