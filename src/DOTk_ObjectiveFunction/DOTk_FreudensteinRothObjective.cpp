/*
 * DOTk_FreudensteinRothObjective.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_FreudensteinRothObjective.hpp"

namespace dotk
{

DOTk_FreudensteinRothObjective::DOTk_FreudensteinRothObjective()
{
}

DOTk_FreudensteinRothObjective::~DOTk_FreudensteinRothObjective()
{
}

Real DOTk_FreudensteinRothObjective::value(const dotk::vector<Real> & primal_)
{
    Real f1 = static_cast<Real>(-13.) + primal_[0]
            + ((static_cast<Real>(5.) - primal_[1]) * primal_[1] - static_cast<Real>(2.)) * primal_[1];
    Real f2 = static_cast<Real>(-29.) + primal_[0]
            + ((primal_[1] + static_cast<Real>(1.)) * primal_[1] - static_cast<Real>(14.)) * primal_[1];
    Real value = f1 * f1 + f2 * f2;

    return (value);
}

void DOTk_FreudensteinRothObjective::gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_)
{
    Real f1 = static_cast<Real>(-13.) + primal_[0]
            + ((static_cast<Real>(5.) - primal_[1]) * primal_[1] - static_cast<Real>(2.)) * primal_[1];
    Real f2 = static_cast<Real>(-29.) + primal_[0]
            + ((primal_[1] + static_cast<Real>(1.)) * primal_[1] - static_cast<Real>(14.)) * primal_[1];

    Real df11 = 1.;
    Real df12 = static_cast<Real>(10.) * primal_[1] - static_cast<Real>(3.) * primal_[1] * primal_[1]
            - static_cast<Real>(2.);
    Real df21 = 1.;
    Real df22 = static_cast<Real>(3.) * primal_[1] * primal_[1] + static_cast<Real>(2.) * primal_[1]
            - static_cast<Real>(14.);

    output_[0] = static_cast<Real>(2.) * (df11 * f1 + df21 * f2);
    output_[1] = static_cast<Real>(2.) * (df12 * f1 + df22 * f2);
}

void DOTk_FreudensteinRothObjective::hessian(const dotk::vector<Real> & primal_,
                                             const dotk::vector<Real> & vector_,
                                             dotk::vector<Real> & output_)
{
    Real f1 = static_cast<Real>(-13.) + primal_[0]
            + ((static_cast<Real>(5.) - primal_[1]) * primal_[1] - static_cast<Real>(2.)) * primal_[1];
    Real f2 = static_cast<Real>(-29.) + primal_[0]
            + ((primal_[1] + static_cast<Real>(1.)) * primal_[1] - static_cast<Real>(14.)) * primal_[1];

    Real df11 = 1.;
    Real df12 = static_cast<Real>(10.) * primal_[1] - static_cast<Real>(3.) * primal_[1] * primal_[1]
            - static_cast<Real>(2.);
    Real df21 = 1.;
    Real df22 = static_cast<Real>(3.) * primal_[1] * primal_[1] + static_cast<Real>(2.) * primal_[1]
            - static_cast<Real>(14.);

    Real df1_22 = static_cast<Real>(10.) - static_cast<Real>(6.) * primal_[1];
    Real df2_22 = static_cast<Real>(6.) * primal_[1] + static_cast<Real>(2.);

    Real h11 = static_cast<Real>(2.) * (df11 * df11) + static_cast<Real>(2.) * (df21 * df21);
    Real h12 = static_cast<Real>(2.) * (df12 * df11) + static_cast<Real>(2.) * (df22 * df21);
    Real h22 = static_cast<Real>(2.) * (df1_22 * f1 + df12 * df12) + 2. * (df2_22 * f2 + df22 * df22);

    output_[0] = h11 * vector_[0] + h12 * vector_[1];
    output_[1] = h12 * vector_[0] + h22 * vector_[1];
}

}
