#ifndef __LIBRARY_HPP__
#define __LIBRARY_HPP__

#include "exception.hpp"

namespace library 
{
    template<class TValue>
    class Dataset
    {
    public:
        Dataset(const std::map<std::string, TValue>& m)
            : m_map(m.begin(), m.end())
        { }

        void dump(std::ostream& os) const 
        {
            for (const auto& m : m_map)
                os << m.first << ": " << m.second << "\n";
        }

        inline const TValue& get(const std::string& sKey) const
        {
            auto pPair = m_map.find(sKey);
            if(pPair == m_map.end())
                THROW(std::runtime_error, std::string("Value for key ") + sKey + " not found!");
            return pPair->second;
        }

        inline void overwrite(const std::string& sKey, const TValue& newval)
        {
            m_map.insert_or_assign(sKey, newval);
//            auto res = m_map.insert(std::make_pair(sKey, newval));						
//            if(!res.second) 
//            {
                // the element was not inserted (because it already exists)
//                res.first->second = newval;
//            }
        }

    private:
        struct Compare 
            //: public std::binary_function<std::string, std::string, bool> 
        {
            struct nocase_compare 
                //: public std::binary_function<unsigned char, unsigned char, bool> 
            {
                bool operator() (const unsigned char& c1,  const unsigned char& c2) const 
                {
                    return std::tolower(c1) < std::tolower(c2);
                }
            };

            bool operator() (const std::string& s1, const std::string& s2) const 
            {
                return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(), nocase_compare());
            }
        };

        std::map<std::string, TValue, Compare> m_map;
    };

    extern Dataset<double> Charges;

    // gives molar mass in g/mol
    extern Dataset<double> Masses;

    // translates an element number to respective symbol string
    extern std::map<size_t, std::string> ElementNo2Symbol;
}


#endif
