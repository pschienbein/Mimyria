#ifndef __PLUGIN_HPP__
#define __PLUGIN_HPP__

#include "pch.hpp"
#include "configfile.hpp"
#include "factory.hpp"

class Plugins
{
public:
    static void Load(const std::string& name);
    static void Load(ConfigFilePtr pCF);

private:
    static void RegisterMany(const registry::FactoryEntry* pList, size_t count);

    class Plugin 
    {
    public:
        Plugin();
        ~Plugin();

        void load(const std::string& sName);

    private:
        void* m_pHandle;
    };
    std::vector<std::shared_ptr<Plugin>> m_aPlugins;
};

#endif 
