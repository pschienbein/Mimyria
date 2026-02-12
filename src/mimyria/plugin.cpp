
#include <dlfcn.h>
#include "plugin.hpp" 
#include "task.hpp"

using namespace std;

void Plugins::Load(const std::string& sName)
{
    static Plugins inst;     
    auto pP = make_shared<Plugin>();
    pP->load(sName);
    inst.m_aPlugins.push_back(pP);
}

void Plugins::Load(ConfigFilePtr pCF)
{
    auto sections = pCF->getSections("plugin");
    for (auto plugin_sec = sections.first; plugin_sec != sections.second; ++plugin_sec)
    {
        string sName = pCF->get(plugin_sec, "name");

        try 
        {
            Plugins::Load(sName);
            cerr << "# Successfully loaded Plugin " << sName << endl;
        }
        catch(runtime_error& e)
        {
            cerr << "# FAILED to load Plugin " << sName << " (" << e.what() << ")\n";
        }
    }
}

Plugins::Plugin::Plugin()
    : m_pHandle(nullptr)
{
}

Plugins::Plugin::~Plugin()
{
    if(m_pHandle)
        dlclose(m_pHandle);
    m_pHandle = nullptr;
}

void Plugins::Plugin::load(const string& sName)
{
    m_pHandle = dlopen(sName.c_str(), RTLD_LAZY);
    if(!m_pHandle)
        THROW(runtime_error, "Plugin cannot be loaded:", dlerror());

    //void (*plugin_register)();
    //plugin_register = (void (*)())dlsym(m_pHandle, "plugin_register");
    registry::plugin_register_fn plugin_register = reinterpret_cast<registry::plugin_register_fn>(dlsym(m_pHandle, "plugin_register"));
    if(!plugin_register)
        THROW(runtime_error, "Symbol \"plugin_register\" could not be imported:", dlerror());

    registry::RegistrationAPI reg_api = { &Plugins::RegisterMany };
    plugin_register(&reg_api);
}

void Plugins::RegisterMany(const registry::FactoryEntry* pList, size_t count)
{
    for(size_t i = 0; i < count; ++i)
    {
        auto& entry = pList[i];
        if(strcmp(entry.factory, "task") == 0)
        {
            auto fn = reinterpret_cast<registry::TaskFactory::CreateInterfaceFn>(entry.pfCreate);
            registry::TaskFactory::Get().Register(entry.name, fn);
        }
    }
}
