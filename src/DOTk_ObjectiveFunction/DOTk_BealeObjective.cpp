/*
 * DOTk_BealeObjective.cpp
 *
 *  Created on: May 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_BealeObjective.hpp"

namespace dotk
{

DOTk_BealeObjective::DOTk_BealeObjective()
{
}

DOTk_BealeObjective::~DOTk_BealeObjective()
{
}

Real DOTk_BealeObjective::value(const dotk::vector<Real> & primal_)
{
    Real f1 = static_cast<Real>(1.5) - primal_[0] * (1.0 - primal_[1]);
    Real f2 = static_cast<Real>(2.25) - primal_[0] * (static_cast<Real>(1.0) - std::pow(primal_[1], 2.));
    Real f3 = static_cast<Real>(2.625) - primal_[0] * (static_cast<Real>(1.0) - std::pow(primal_[1], 3.));

    Real value = std::pow(f1, 2.) + std::pow(f2, 2.) + std::pow(f3, 2.);

    return (value);
}

void DOTk_BealeObjective::gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & derivative_)
{
    Real f1 = static_cast<Real>(1.5) - primal_[0] * (static_cast<Real>(1.0) - primal_[1]);
    Real f2 = static_cast<Real>(2.25) - primal_[0] * (static_cast<Real>(1.0) - pow(primal_[1], 2.));
    Real f3 = static_cast<Real>(2.625) - primal_[0] * (static_cast<Real>(1.0) - pow(primal_[1], 3.));

    Real df1dx = -(static_cast<Real>(1.0) - primal_[1]);
    Real df1dy = primal_[0];
    Real df2dx = -(static_cast<Real>(1.0) - std::pow(primal_[1], 2.));
    Real df2dy = static_cast<Real>(2.0) * primal_[0] * primal_[1];
    Real df3dx = -(static_cast<Real>(1.0) - std::pow(primal_[1], 3.));
    Real df3dy = static_cast<Real>(3.0) * primal_[0] * std::pow(primal_[1], 2.);

    derivative_[0] = static_cast<Real>(2.0) * df1dx * f1 + static_cast<Real>(2.0) * df2dx * f2
            + static_cast<Real>(2.0) * df3dx * f3;
    derivative_[1] = static_cast<Real>(2.0) * df1dy * f1 + static_cast<Real>(2.0) * df2dy * f2
            + static_cast<Real>(2.0) * df3dy * f3;
}

void DOTk_BealeObjective::hessian(const dotk::vector<Real> & primal_,
                                  const dotk::vector<Real> & delta_primal_,
                                  dotk::vector<Real> & hessian_times_delta_primal_)
{
    Real f1 = static_cast<Real>(1.5) - primal_[0] * (static_cast<Real>(1.0) - primal_[1]);
    Real f2 = static_cast<Real>(2.25) - primal_[0] * (static_cast<Real>(1.0) - std::pow(primal_[1], 2.));
    Real f3 = static_cast<Real>(2.625) - primal_[0] * (static_cast<Real>(1.0) - std::pow(primal_[1], 3.));

    Real df1dx = -(static_cast<Real>(1.0) - primal_[1]);
    Real df1dy = primal_[0];
    Real df2dx = -(static_cast<Real>(1.0) - std::pow(primal_[1], 2.));
    Real df2dy = static_cast<Real>(2.0) * primal_[0] * primal_[1];
    Real df3dx = -(static_cast<Real>(1.0) - std::pow(primal_[1], 3.));
    Real df3dy = static_cast<Real>(3.0) * primal_[0] * std::pow(primal_[1], 2.);
    Real d2f1dx2 = 0.;
    Real d2f1dy2 = 0.;
    Real d2f1dxdy = 1.0;
    Real d2f2dx2 = 0.;
    Real d2f2dy2 = static_cast<Real>(2.0) * primal_[0];
    Real d2f2dxdy = static_cast<Real>(2.0) * primal_[1];
    Real d2f3dx2 = 0.;
    Real d2f3dy2 = static_cast<Real>(6.0) * primal_[0] * primal_[1];
    Real d2f3dxdy = static_cast<Real>(3.0) * std::pow(primal_[1], 2.);

    Real H11 = static_cast<Real>(2.) * (d2f1dx2 * f1 + df1dx * df1dx)
            + static_cast<Real>(2.) * (d2f2dx2 * f2 + df2dx * df2dx)
            + static_cast<Real>(2.) * (d2f3dx2 * f3 + df3dx * df3dx);
    Real H22 = static_cast<Real>(2.) * (d2f1dy2 * f1 + df1dy * df1dy)
            + static_cast<Real>(2.) * (d2f2dy2 * f2 + df2dy * df2dy)
            + static_cast<Real>(2.) * (d2f3dy2 * f3 + df3dy * df3dy);
    Real H12 = static_cast<Real>(2.) * (d2f1dxdy * f1 + df1dx * df1dy)
            + static_cast<Real>(2.) * (d2f2dxdy * f2 + df2dx * df2dy)
            + static_cast<Real>(2.) * (d2f3dxdy * f3 + df3dx * df3dy);

    hessian_times_delta_primal_[0] = H11 * delta_primal_[0] + H12 * delta_primal_[1];
    hessian_times_delta_primal_[1] = H12 * delta_primal_[0] + H22 * delta_primal_[1];
}

}
