/*
 * DOTk_PowellObjective.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "DOTk_PowellObjective.hpp"

namespace dotk
{

DOTk_PowellObjective::DOTk_PowellObjective()
{
}

DOTk_PowellObjective::~DOTk_PowellObjective()
{
}

Real DOTk_PowellObjective::value(const dotk::Vector<Real> & primal_)
{
    Real f1 = static_cast<Real>(1e4) * primal_[0] * primal_[1] - static_cast<Real>(1.);
    Real f2 = std::exp(-primal_[0]) + std::exp(-primal_[1]) - static_cast<Real>(1.0001);
    Real value = f1 * f1 + f2 * f2;

    return (value);
}

void DOTk_PowellObjective::gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_)
{
    Real f1 = static_cast<Real>(1e4) * primal_[0] * primal_[1] - static_cast<Real>(1.);
    Real f2 = std::exp(-primal_[0]) + std::exp(-primal_[1]) - static_cast<Real>(1.0001);

    Real f11 = static_cast<Real>(1e4) * primal_[1];
    Real f12 = static_cast<Real>(1e4) * primal_[0];
    Real f21 = -std::exp(-primal_[0]);
    Real f22 = -std::exp(-primal_[1]);

    output_[0] = static_cast<Real>(2.) * (f11 * f1 + f21 * f2);
    output_[1] = static_cast<Real>(2.) * (f12 * f1 + f22 * f2);
}

void DOTk_PowellObjective::hessian(const dotk::Vector<Real> & primal_,
                                   const dotk::Vector<Real> & vector_,
                                   dotk::Vector<Real> & output_)
{
    Real f1 = static_cast<Real>(1e4) * primal_[0] * primal_[1] - static_cast<Real>(1.);
    Real f2 = std::exp(-primal_[0]) + std::exp(-primal_[1]) - static_cast<Real>(1.0001);

    Real f11 = 1e4 * primal_[1];
    Real f12 = 1e4 * primal_[0];
    Real f21 = -std::exp(-primal_[0]);
    Real f22 = -std::exp(-primal_[1]);

    Real f111 = 0.;
    Real f112 = 1e4;
    Real f121 = 1e4;
    Real f122 = 0.;
    Real f211 = std::exp(-primal_[0]);
    Real f212 = 0.;
    Real f221 = 0.;
    Real f222 = std::exp(-primal_[1]);

    Real h11 = static_cast<Real>(2.) * (f111 * f1 + f11 * f11) + static_cast<Real>(2.) * (f211 * f2 + f21 * f21);
    Real h12 = static_cast<Real>(2.) * (f112 * f1 + f11 * f12) + static_cast<Real>(2.) * (f212 * f2 + f21 * f22);
    Real h21 = static_cast<Real>(2.) * (f121 * f1 + f21 * f11) + static_cast<Real>(2.) * (f221 * f2 + f22 * f21);
    Real h22 = static_cast<Real>(2.) * (f122 * f1 + f12 * f12) + static_cast<Real>(2.) * (f222 * f2 + f22 * f22);

    output_[0] = h11 * vector_[0] + h12 * vector_[1];
    output_[1] = h21 * vector_[0] + h22 * vector_[1];
}

}
