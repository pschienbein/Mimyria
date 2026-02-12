#ifndef __EXCEPTION_HPP__
#define __EXCEPTION_HPP__

/////////////////////////////////////////////////////////
// Trim a string
inline void ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
}

inline void rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), s.end());
}

inline void trim(std::string& s)
{
    ltrim(s);
    rtrim(s);
}

/////////////////////////////////////////////////////////
// Emulating sprintf
template<typename T>
void appendToStream(std::ostringstream& oss, const T& last)
{
    oss << last;
}

template<typename T, typename... Args>
void appendToStream(std::ostringstream& oss, const T& first, const Args&... args)
{
    oss << first;
    appendToStream(oss, args...);
}

template<typename... Args>
std::string sprint(const Args&... args) 
{
    std::ostringstream oss;
    appendToStream(oss, args...);
    return oss.str();
}


#define THROW(ex, ...) throw ex(sprint(__VA_ARGS__, "[", __FILE__, ":", __LINE__, "]"))
#define THROW_WITH_NESTED(ex, ...) std::throw_with_nested(ex(sprint(__VA_ARGS__, "[", __FILE__, ":", std::to_string(__LINE__), "]")))
        
//#define THROW(ex, str) throw ex(std::string(str) + "[" + __FILE__ + ":" + std::to_string(__LINE__) + "]")
//#define THROW_WITH_NESTED(ex, str) std::throw_with_nested(ex(std::string(str) + "[" + __FILE__ + ":" + std::to_string(__LINE__) + "]"))

// This function prints an exception to file, including all nested exceptions (if any)
void print_nested_exception(std::ostream& os, const std::exception& e, int level = 0);
void print_nested_exception(std::ostream& os, const std::exception_ptr& e);

#endif
