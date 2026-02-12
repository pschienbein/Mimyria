#ifndef __FACTORY_HPP_
#define __FACTORY_HPP_

#include "exception.hpp"

namespace registry
{

    template<class Interface>
    class Factory
    {
    public:
        typedef std::shared_ptr<Interface> InterfacePtr;
        typedef InterfacePtr (*CreateInterfaceFn)();
        typedef std::map<std::string, CreateInterfaceFn> CreatorMap;

        // Default function Creator:
        template<class TChildInterface>
        static inline InterfacePtr DefaultCreator() { return InterfacePtr(new TChildInterface); }

        static Factory<Interface>& Get()
        {
            static Factory<Interface> factory;
            return factory;
        }

        inline InterfacePtr Create(const std::string& sName)
        {
            auto this_type = typeid(Interface).name();

            auto pCreator = m_mCreators.find(sName);
            if(pCreator == m_mCreators.end())
                THROW(std::runtime_error, std::string("Creator ") + sName + " not found for interface " + this_type);
            return (*pCreator->second)();
        }

        inline void Dump()
        {
            std::cerr << "Registered Creators: \n";
            for(const auto& p : m_mCreators)
                std::cerr << p.first << std::endl;
        }

        inline void Register(const std::string& sName, CreateInterfaceFn pfCreator)
        {
            if(m_mCreators.find(sName) != m_mCreators.end())
                throw std::runtime_error("Creator alrady defined");
            m_mCreators.insert(std::make_pair(sName, pfCreator));
        }

    private:
        CreatorMap m_mCreators;
    };


    // API for handling plugins
    typedef void* CreateFnErased;

    struct FactoryEntry
    {
        const char* factory;
        const char* name;
        CreateFnErased pfCreate;
    };

    struct RegistrationAPI 
    {
        void (*register_many)(const FactoryEntry* pEntries, size_t count);
    };

    typedef void (*plugin_register_fn)(const RegistrationAPI* pAPI);
}


#define CONCAT_(x,y) x##y
#define CONCAT(x,y) CONCAT_(x,y)

// Start of macro REGISTER
#define REGISTER(factory, name, cls)  \
namespace \
{ \
    static struct CONCAT(Registerer, __LINE__) \
    { \
        CONCAT(Registerer, __LINE__)() \
        { \
            factory::Get().Register(name, factory::DefaultCreator<cls>); \
        } \
    } \
    CONCAT(reg, __LINE__); \
} 
// End of macro REGISTER

// Start of macro REGISTER_DIRECT
#define REGISTER_DIRECT(factory, name, cls) \
factory::Get().Register(name, factory::DefaultCreator<cls>);

#endif
