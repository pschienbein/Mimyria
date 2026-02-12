#include "pch.hpp"
#include "box.hpp"

using namespace std;

Box::Box() 
    : m_bDiagonalMatrix(false)
{
}

Box::~Box() 
{}

// transforms absolute cartesian coordinates into fractional cell coordinates
// NOTE: WITHOUT applying periodic boundary conditions, which is trivial by just taking the decimal part of the number
Vector3d Box::abs2frac(const Vector3d& v) const 
{
    if(m_bDiagonalMatrix)
        return Vector3d(v.x() / m_vA.x(), v.y() / m_vB.y(), v.z() / m_vC.z());
    else 
        return m_mCartesian2Cell * v;
}

// transforms fractional cell coordinates into absolute cartesian coordinates
Vector3d Box::frac2abs(const Vector3d& v) const 
{
    if(m_bDiagonalMatrix)
        return Vector3d(v.x() * m_vA.x(), v.y() * m_vB.y(), v.z() * m_vC.z());
    else 
        return m_mCell2Cartesian * v;
}

// wraps a vector into the box including periodic boundary conditions
Vector3d Box::wrap(const Vector3d& v) const
{
    auto vFrac = abs2frac(v);
    vFrac.x() -= static_cast<long>(vFrac.x());
    vFrac.y() -= static_cast<long>(vFrac.y());
    vFrac.z() -= static_cast<long>(vFrac.z());
    return frac2abs(vFrac);
}

// setupializes the box using the three box vectors
void Box::setup(const Vector3d& a, const Vector3d& b, const Vector3d& c)
{
    // copy the vectors
    m_vA = a; 
    m_vB = b;
    m_vC = c;

    // test if the vectors form a diagonal matrix, i.e. a cubic or orthorhombic box
    // This is a test _exactly_ for zero which should only happen, if the vectors are manually set to zero.
    if(m_vA.y() == 0 && m_vA.z() == 0 && m_vB.x() == 0 && m_vB.z() == 0 && m_vC.x() == 0 && m_vC.y() == 0)
        m_bDiagonalMatrix = true;
    else 
    {
        // the matrix is not diagonal
        m_bDiagonalMatrix = false;            

        // Create the associated matrices
        m_mCell2Cartesian.col(0) = m_vA;
        m_mCell2Cartesian.col(1) = m_vB;
        m_mCell2Cartesian.col(2) = m_vC;
        m_mCartesian2Cell = m_mCell2Cartesian.inverse();

        m_mCell2CartesianAbs.col(0) = m_vA.normalized();
        m_mCell2CartesianAbs.col(1) = m_vB.normalized();
        m_mCell2CartesianAbs.col(2) = m_vC.normalized();
        m_mCartesian2CellAbs = m_mCell2CartesianAbs.inverse();
    }
}

void Box::setup(ConfigFilePtr pCF)
{
    auto pCellSection = pCF->getSections("cell").first;

    // for now, just the simplest definition
    setup(Vector3d(
                pCF->as<double>(pCellSection, "a1"),
                pCF->as<double>(pCellSection, "a2"),
                pCF->as<double>(pCellSection, "a3")
                ),
            Vector3d(
                pCF->as<double>(pCellSection, "b1"),
                pCF->as<double>(pCellSection, "b2"),
                pCF->as<double>(pCellSection, "b3")
                ),
            Vector3d(
                pCF->as<double>(pCellSection, "c1"),
                pCF->as<double>(pCellSection, "c2"),
                pCF->as<double>(pCellSection, "c3")
                )
         );
}

// Wendet die PBC auf einen Differenzvektor an!
Vector3d Box::pbc(const Vector3d& v) 
{
    // if this is a diagonal matrix, the process is easier!
    if(m_bDiagonalMatrix)
    {
         Vector3d tmp(
                v.x() - m_vA.x() * nearbyint(v.x() / m_vA.x()),
                v.y() - m_vB.y() * nearbyint(v.y() / m_vB.y()),
                v.z() - m_vC.z() * nearbyint(v.z() / m_vC.z())
                    );
         return tmp;
    }
    else 
    {
        // this transformation transforms the cartesian coordinates into fractional coordinates along the a,b, and c axes! => range [0,1]
        Vector3d vTransformed = m_mCartesian2Cell * v;
        // now remove the pbc!
        Vector3d vPBC(nearbyint(vTransformed.x()), nearbyint(vTransformed.y()), nearbyint(vTransformed.z()));
        vTransformed -= vPBC;
        // transform back into absolute cartesian coordinates
        return m_mCell2Cartesian * vTransformed;
    }
}

// Gibt die drei Raumvektoren zurück, die die Box aufspannen
void Box::vectors(Vector3d& a, Vector3d& b, Vector3d& c) const 
{
    // just copy the three vectors
    a = m_vA; 
    b = m_vB; 
    c = m_vC;
}

// Berechnet das Volumen der Box in m^3
double Box::volume() const 
{
    // just a convenient method to compute the volume right from this interface
    return m_vA.dot(m_vB.cross(m_vC));
}
