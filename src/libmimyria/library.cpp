#include "pch.hpp"
#include "library.hpp"

using namespace std;

namespace library 
{
    // initialize the charges array with the default Wannier charges. 
    // more charges can be set via the overwrite function and 
    // charges can also overwritten using that method
    map<string,double> mCharges = {
        {"H", 1},
        {"O", 6},
        {"X", -2}
    };
    Dataset<double> Charges(mCharges);

    // initialize the mass array  
    // more masses can be set via the overwrite function and 
    // they can also overwritten using that method
    // this is g/mol
    map<string, double> mMasses = {
        {"H", 1.0079},
        {"C", 12.011},
        {"N", 14.0067},
        {"O", 15.999},
        {"F", 18.9984},
        {"S", 32.065},
        {"V", 50.9415},
        {"Fe", 55.845},
        {"Bi", 208.9804}
    };
    Dataset<double> Masses(mMasses);


    std::map<size_t, std::string> ElementNo2Symbol = {
        {1, "H"}, 
        {6, "C"},
        {7, "N"},
        {8, "O"},
        {23, "V"},
        {83, "Bi"}
    };
}
