#ifndef __BOX_HPP__
#define __BOX_HPP__ 

#include "pch.hpp"
#include "configfile.hpp"

using Eigen::Vector3d;
using Eigen::Matrix3d;

class Box 
{
public:		
    Box();
    //Box(const Box& rhs);
    ~Box();

    // transforms absolute cartesian coordinates into fractional cell coordinates
    // NOTE: WITHOUT applying periodic boundary conditions, which is trivial by just taking the decimal part of the number
    Vector3d abs2frac(const Vector3d& v) const;
    // transforms fractional cell coordinates into absolute cartesian coordinates
    Vector3d frac2abs(const Vector3d& v) const;

    // transforms cartesian coordinates to lattice coordinates
    inline Vector3d cartesian2cell(const Vector3d& v) const
    {
        return m_mCartesian2CellAbs * v;
    }
    // transforms lattice coordinates to cartesian coordinates
    inline Vector3d cell2cartesian(const Vector3d& v) const 
    {
        return m_mCell2CartesianAbs * v;
    }

    // wraps a vector into the box including periodic boundary conditions
    Vector3d wrap(const Vector3d& v) const;

    // initializes the box using the three box vectors
    void setup(const Vector3d& a, const Vector3d& b, const Vector3d& c);
    // initializes the box using a section of the control file
    void setup(ConfigFilePtr pCF);

    // Wendet die PBC auf einen Differenzvektor an!
    Vector3d pbc(const Vector3d& vDiff);
    // Gibt die drei Raumvektoren zurück, die die Box aufspannen
    void vectors(Vector3d& a, Vector3d& b, Vector3d& c) const;
    // Berechnet das Volumen der Box in m^3
    double volume() const;

private:
    // the three box vectors
    Vector3d m_vA, m_vB, m_vC;
    // Matrices used for the rotation between the cartesian coordinates and the FRACTIONAL cell coordinates
    Matrix3d m_mCartesian2Cell, m_mCell2Cartesian;
    // true, if the three vectors form a diagonal matrix, i.e. a cubic or orthorhombic box whose vectors are aligned along the x,y,z cartesian coordinates
    bool m_bDiagonalMatrix;
    // Matrices used for the rotation between the cartesian coordinates and the ABSOLUTE cell coordinates
    Matrix3d m_mCartesian2CellAbs, m_mCell2CartesianAbs;
};

typedef std::shared_ptr<Box> BoxPtr;

#endif
