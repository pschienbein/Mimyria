#ifndef __CONFIGFILE_HPP__
#define __CONFIGFILE_HPP__

#include "exception.hpp"


class ConfigFile 
{
public:
    typedef std::map<std::string, std::string> KVMap;
    typedef std::unordered_multimap<std::string, KVMap> DataMap;
    typedef DataMap::const_iterator SectionPtr;
    typedef std::pair<SectionPtr, SectionPtr> SectionRange;

    void open(const std::string& filename);

    // loop over sections
    // if no search string is given, all sections are returned
    SectionRange getSections(const std::string& sSectionName = "") const;

    // returns the section with the given name; 
    // throws exception if section is occurring multiple times or not present
    SectionPtr getSection(const std::string& sSectionName) const;

    bool isSectionPresent(const std::string& section) const;
    bool isKeyPresent(SectionPtr pSection, const std::string& key) const;

    std::string get(SectionPtr pSection, const std::string& key) const;

    // returns all keys present in a given section
    std::vector<std::string> keys(SectionPtr pSection) const;

    template<typename T>
    T as(SectionPtr pSection, const std::string& key) const
    {
        T convertedValue;
        std::istringstream iss(get(pSection, key));
        if(!(iss >> convertedValue) || !iss.eof())
            THROW(std::runtime_error, "Conversion to type \"", typeid(T).name(), "\" failed for value \"", get(pSection, key), "\"");
        return convertedValue;
    }

    // as function with a default parameter which is returned upon error
    template<typename T>
    T as(SectionPtr pSection, const std::string& key, const T& defaultValue) const
    {
        try 
        {
            return as<T>(pSection, key);            
        }
        catch(std::runtime_error&)
        {
            return defaultValue;
        }
    }

    // writes all content to stream, mainly for debug purposes
    void write(std::ostream& os) const;

private:
    static const std::regex commentPattern;
    static const std::regex blankLinePattern;
    static const std::regex sectionPattern;
    static const std::regex kvPattern;

    std::unordered_multimap<std::string, std::map<std::string, std::string>> m_data;
};

typedef std::shared_ptr<ConfigFile> ConfigFilePtr; 

#endif
