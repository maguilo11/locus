/*
 * DOTk_GcmmaTestInequalityConstraint.cpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_GcmmaTestInequalityConstraint.hpp"

namespace dotk
{

DOTk_GcmmaTestInequalityConstraint::DOTk_GcmmaTestInequalityConstraint() :
        dotk::DOTk_InequalityConstraint<Real>(),
        m_Constant(1.)
{
}

DOTk_GcmmaTestInequalityConstraint::~DOTk_GcmmaTestInequalityConstraint()
{
}

Real DOTk_GcmmaTestInequalityConstraint::bound()
{
    return (m_Constant);
}

Real DOTk_GcmmaTestInequalityConstraint::value(const dotk::vector<Real> & primal_)
{
    Real term_one = static_cast<Real>(61.) / std::pow(primal_[0], 3.);
    Real term_two = static_cast<Real>(37.) / std::pow(primal_[1], 3.);
    Real term_three = static_cast<Real>(19.) / std::pow(primal_[2], 3.);
    Real term_four = static_cast<Real>(7.) / std::pow(primal_[3], 3.);
    Real term_five = static_cast<Real>(1.) / std::pow(primal_[4], 3.);

    Real sum = term_one + term_two + term_three + term_four + term_five;

    return (sum);
}

Real DOTk_GcmmaTestInequalityConstraint::residual(const dotk::vector<Real> & primal_)
{
    Real current_residual = this->value(primal_) - this->bound();
    return (current_residual);
}

void DOTk_GcmmaTestInequalityConstraint::gradient(const dotk::vector<Real> & primal_,
                                                  dotk::vector<Real> & gradient_)
{
    Real factor = -3.;

    gradient_[0] = factor * (static_cast<Real>(61.) / std::pow(primal_[0], 4.));
    gradient_[1] = factor * (static_cast<Real>(37.) / std::pow(primal_[1], 4.));
    gradient_[2] = factor * (static_cast<Real>(19.) / std::pow(primal_[2], 4.));
    gradient_[3] = factor * (static_cast<Real>(7.) / std::pow(primal_[3], 4.));
    gradient_[4] = factor * (static_cast<Real>(1.) / std::pow(primal_[4], 4.));
}

}
