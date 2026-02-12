#include "pch.hpp"
#include "configfile.hpp" 
#include "exception.hpp"

using namespace std;

const regex ConfigFile::commentPattern(R"(\s*[;#].*)");
const regex ConfigFile::sectionPattern(R"(\s*\[(.+?)\]\s*)");
const regex ConfigFile::blankLinePattern(R"(^\s*$)");
const regex ConfigFile::kvPattern(R"(\s*([^=:#]+?)\s*[=:]\s*(.*?)\s*)");

void ConfigFile::open(const string& filename)
{
    ifstream ifs(filename);
    if(!ifs.is_open())
        THROW(runtime_error, "Unable to open file configuration file \"", filename, "\"");

    string line;
    auto pCurrentSection = m_data.end();
    
    while(getline(ifs, line))
    {
        line = regex_replace(line, regex(R"(^\s+|\s+$)"), "");

        // Skip comment lines and blank lines
        if(regex_match(line, commentPattern) || regex_match(line, blankLinePattern))
            continue;

        smatch sectionMatch;
        if(regex_match(line, sectionMatch, sectionPattern))
        {
            pCurrentSection = m_data.insert(make_pair(sectionMatch[1], map<string, string>()));
            continue;
        }

        smatch kvMatch;
        if(regex_match(line, kvMatch, kvPattern))
        {
            if(pCurrentSection != m_data.end())
                pCurrentSection->second[kvMatch[1]] = kvMatch[2];
        }
    }
}

bool ConfigFile::isSectionPresent(const std::string& section) const
{
    return m_data.find(section) != m_data.end();
}

bool ConfigFile::isKeyPresent(SectionPtr pSection, const std::string& key) const 
{
    return pSection->second.find(key) != pSection->second.end();
}

std::string ConfigFile::get(SectionPtr pSection, const std::string& key) const
{
    auto keyIt = pSection->second.find(key);
    if(keyIt == pSection->second.end())
        THROW(runtime_error, "Key \"", key, "\" not present in section [", pSection->first, "]");

    return keyIt->second;
}

ConfigFile::SectionPtr ConfigFile::getSection(const std::string& sSectionName) const
{
    auto range = getSections(sSectionName);

    if(range.first == range.second)
        THROW(runtime_error, "Section \"", sSectionName, "\" not found in ConfigFile, but requested");

    if (next(range.first) != range.second)
        THROW(runtime_error, "Section \"", sSectionName, "\" occurs more than once in ConfigFile");

    return range.first;
}

ConfigFile::SectionRange ConfigFile::getSections(const std::string& sSectionName) const 
{
    if(sSectionName.empty())
        return make_pair(m_data.begin(), m_data.end());
    else
        return m_data.equal_range(sSectionName);
}

std::vector<std::string> ConfigFile::keys(SectionPtr pSection) const 
{
    vector<string> keys; 
    keys.reserve(pSection->second.size());

    for(const auto& [key, value] : pSection->second)
        keys.push_back(key);

    return keys;
}

void ConfigFile::write(ostream& os) const 
{
    for(auto p : m_data)
    {
    //std::unordered_map<std::string, std::map<std::string, std::string>> m_data;
        os << "[" << p.first << "]\n";
        for(auto kv : p.second)
            os << kv.first << " = " << kv.second << "\n";
    }
}

