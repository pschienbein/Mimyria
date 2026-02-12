
#include "pch.hpp"
#include "exception.hpp" 

#ifdef __GNUG__
  #include <cxxabi.h>
#endif

using namespace std;

string demangled_type_name(const std::type_info& ti)
{
#ifdef __GNUG__
    int status = 0;
    unique_ptr<char, void(*)(void*)> buf {
        abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status), std::free
    };

    if(status == 0 && buf)
        return string(buf.get());
#endif

    return string(ti.name());
}

void print_nested_exception(std::ostream& os, const std::exception& e, int level)
{
    os << std::string(2+2*level, ' ') 
       << "EXCEPTION [" 
       << demangled_type_name(typeid(e)) << "]: "
       << e.what() << '\n';
    try 
    {
        std::rethrow_if_nested(e);
    } 
    catch(const std::exception& e) 
    {
        print_nested_exception(os, e, level+1);
    } 
    catch(...) 
    {}
}
void print_nested_exception(std::ostream& os, const std::exception_ptr& eptr)
{
    try 
    {
        if(eptr)
            std::rethrow_exception(eptr);
    }
    catch(exception& e)
    {
        print_nested_exception(os, e);
    }
}
