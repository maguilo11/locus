/*
 * DOTk_NocedalAndWrightEquality.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_NocedalAndWrightEquality.hpp"

namespace dotk
{

DOTk_NocedalAndWrightEquality::DOTk_NocedalAndWrightEquality() :
        dotk::DOTk_EqualityConstraint<Real>()
{
}

DOTk_NocedalAndWrightEquality::~DOTk_NocedalAndWrightEquality()
{
}

void DOTk_NocedalAndWrightEquality::residual(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_)
{
    // C1 = u1^2 + u2^2 + u3^2 + u4^2 + u5^2 - 10
    output_[0] = pow(primal_[0], static_cast<Real>(2.)) + pow(primal_[1], static_cast<Real>(2.))
            + pow(primal_[2], static_cast<Real>(2.)) + pow(primal_[3], 2.) + pow(primal_[4], 2.)
            - static_cast<Real>(10.);
    // C2 = u2 * u3 - 5. * u4 * u5
    output_[1] = primal_[1] * primal_[2] - static_cast<Real>(5.) * primal_[3] * primal_[4];
    // C3 = u1^3 + u2^3 + 1.
    output_[2] = pow(primal_[0], static_cast<Real>(3.)) + pow(primal_[1], static_cast<Real>(3.))
            + static_cast<Real>(1.);
}

void DOTk_NocedalAndWrightEquality::jacobian(const dotk::vector<Real> & primal_,
                                             const dotk::vector<Real> & vector_,
                                             dotk::vector<Real> & output_)
{
    // |  2*u1    2*u2   2*u3   2*u4   2*u5 |
    // |   0       u3     u2   -5*u5  -5*u4 |
    // | 3*u1^2  3*u2^2   0      0      0   |
    output_[0] = static_cast<Real>(2.) * primal_[0] * vector_[0] + static_cast<Real>(2.) * primal_[1] * vector_[1]
            + static_cast<Real>(2.) * primal_[2] * vector_[2] + static_cast<Real>(2.) * primal_[3] * vector_[3]
            + static_cast<Real>(2.) * primal_[4] * vector_[4];
    output_[1] = primal_[2] * vector_[1] + primal_[1] * vector_[2] - static_cast<Real>(5.) * primal_[4] * vector_[3]
            - static_cast<Real>(5.) * primal_[3] * vector_[4];
    output_[2] = static_cast<Real>(3.) * pow(primal_[0], 2.) * vector_[0]
            + static_cast<Real>(3.) * pow(primal_[1], static_cast<Real>(2.)) * vector_[1];
}

void DOTk_NocedalAndWrightEquality::adjointJacobian(const dotk::vector<Real> & primal_,
                                                    const dotk::vector<Real> & dual_,
                                                    dotk::vector<Real> & output_)
{
    // | 2*u1    0    3*u1^2 |
    // | 2*u2    u3   3*u2^2 |
    // | 2*u3    u2     0    |
    // | 2*u4  -5*u5    0    |
    // | 2*u5  -5*u4    0    |
    output_[0] = static_cast<Real>(2.) * primal_[0] * dual_[0]
            + static_cast<Real>(3.) * pow(primal_[0], static_cast<Real>(2.)) * dual_[2];
    output_[1] = static_cast<Real>(2.) * primal_[1] * dual_[0] + primal_[2] * dual_[1]
            + static_cast<Real>(3.) * pow(primal_[1], static_cast<Real>(2.)) * dual_[2];
    output_[2] = static_cast<Real>(2.) * primal_[2] * dual_[0] + primal_[1] * dual_[1];
    output_[3] = static_cast<Real>(2.) * primal_[3] * dual_[0] - static_cast<Real>(5.) * primal_[4] * dual_[1];
    output_[4] = static_cast<Real>(2.) * primal_[4] * dual_[0] - static_cast<Real>(5.) * primal_[3] * dual_[1];
}

void DOTk_NocedalAndWrightEquality::hessian(const dotk::vector<Real> & primal_,
                                            const dotk::vector<Real> & dual_,
                                            const dotk::vector<Real> & vector_,
                                            dotk::vector<Real> & output_)
{
    const Real du1 = vector_[0];
    const Real du2 = vector_[1];
    const Real du3 = vector_[2];
    const Real du4 = vector_[3];
    const Real du5 = vector_[4];
    const Real lambda1 = dual_[0];
    const Real lambda2 = dual_[1];
    const Real lambda3 = dual_[2];
    output_[0] = static_cast<Real>(2.) * lambda1 * du1 + static_cast<Real>(6.) * primal_[0] * lambda3 * du1;
    output_[1] = static_cast<Real>(2.) * lambda1 * du2 + static_cast<Real>(6.) * primal_[1] * lambda3 * du2
            + lambda2 * du3;
    output_[2] = lambda2 * du2 + static_cast<Real>(2.) * lambda1 * du3;
    output_[3] = static_cast<Real>(2.) * lambda1 * du4 - static_cast<Real>(5.) * lambda2 * du5;
    output_[4] = static_cast<Real>(-5.) * lambda2 * du4 + static_cast<Real>(2.) * lambda1 * du5;
}

}
