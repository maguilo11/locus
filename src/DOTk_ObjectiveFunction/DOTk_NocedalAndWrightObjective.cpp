/*
 * DOTk_NocedalAndWrightObjective.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_NocedalAndWrightObjective.hpp"

namespace dotk
{

DOTk_NocedalAndWrightObjective::DOTk_NocedalAndWrightObjective()
{
}

DOTk_NocedalAndWrightObjective::~DOTk_NocedalAndWrightObjective()
{
}

Real DOTk_NocedalAndWrightObjective::value(const dotk::vector<Real> & primal_)
{
    // J(X) = exp(u1 * u2 * u3 * u4 * u5) - 0.5 * (1 + u1^3 + u2^3)^2
    const Real a = std::exp(primal_[0] * primal_[1] * primal_[2] * primal_[3] * primal_[4]);
    const Real b = static_cast<Real>(1.) + pow(primal_[0], static_cast<Real>(3.))
            + std::pow(primal_[1], static_cast<Real>(3.));
    const Real c = static_cast<Real>(0.5) * pow(b, 2.);
    Real fval = a - c;
    return (fval);
}

void DOTk_NocedalAndWrightObjective::gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_)
{
    // (1 + x1^3 + x2^3)
    const Real a = 1. + pow(primal_[0], 3) + pow(primal_[1], 3.);
    // exp(x1 * x2 * x3 * x4 * x5)
    const Real b = exp(primal_[0] * primal_[1] * primal_[2] * primal_[3] * primal_[4]);
    // J_x1(X) = -3. * x1^2 * (1 + x1^3 + x2^3) + exp(x1 * x2 * x3 * x4 * x5) * x2 * x3 * x4 * x5
    output_[0] = (-3. * pow(primal_[0], 2.) * a) + (b * primal_[1] * primal_[2] * primal_[3] * primal_[4]);
    // J_x2(X) = -3. * x2^2 * (1 + x1^3 + x2^3) + exp(x1 * x2 * x3 * x4 * x5) * x1 * x3 * x4 * x5
    output_[1] = (-3. * pow(primal_[1], 2.) * a) + (b * primal_[0] * primal_[2] * primal_[3] * primal_[4]);
    // exp(x1 * x2 * x3 * x4 * x5) * x1 * x2 * x4 * x5
    output_[2] = b * primal_[0] * primal_[1] * primal_[3] * primal_[4];
    // exp(x1 * x2 * x3 * x4 * x5) * x1 * x2 * x3 * x5
    output_[3] = b * primal_[0] * primal_[1] * primal_[2] * primal_[4];
    // exp(x1 * x2 * x3 * x4 * x5) * x1 * x2 * x3 * x4
    output_[4] = b * primal_[0] * primal_[1] * primal_[2] * primal_[3];
}

void DOTk_NocedalAndWrightObjective::hessian(const dotk::vector<Real> & primal_,
                                             const dotk::vector<Real> & vector_,
                                             dotk::vector<Real> & output_)
{
    const Real u1 = primal_[0];
    const Real u2 = primal_[1];
    const Real u3 = primal_[2];
    const Real u4 = primal_[3];
    const Real u5 = primal_[4];
    const Real exp_val = exp(u1 * u2 * u3 * u4 * u5);
    // column 1
    // exp(u1 u2 u3 u4 u5) u2^2 u3^2 u4^2 u5^2 - 6 u1 (u1^3 + u2^3 + 1) - 9 u1^4
    const Real H11 = exp_val * pow(u2, static_cast<Real>(2.)) * pow(u3, static_cast<Real>(2.))
            * pow(u4, static_cast<Real>(2.)) * pow(u5, static_cast<Real>(2.))
            - static_cast<Real>(6.) * u1
                    * (pow(u1, static_cast<Real>(3.)) + pow(u2, static_cast<Real>(3.)) + static_cast<Real>(1.))
            - static_cast<Real>(9.) * pow(u1, static_cast<Real>(4.));
    // -9. u1^2 u2^2 + exp(u1 u2 u3 u4 u5) u3 u4 u5 + exp(u1 u2 u3 u4 u5) u1 u2 u3^2 u4^2 u5^2
    const Real H12 = exp_val * u1 * u2 * pow(u3, static_cast<Real>(2.)) * pow(u4, static_cast<Real>(2.))
            * pow(u5, static_cast<Real>(2.)) + u3 * u4 * u5 * exp_val
            - static_cast<Real>(9.) * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.));
    // E^(u1 u2 u3 u4 u5) u2 u4 u5 + E^(u1 u2 u3 u4 u5) u1 u2^2 u3 u4^2 u5^2
    const Real H13 = (exp_val * u2 * u4 * u5)
            + (exp_val * u1 * pow(u2, static_cast<Real>(2.)) * u3 * pow(u4, static_cast<Real>(2.))
                    * pow(u5, static_cast<Real>(2.)));
    // E^(u1 u2 u3 u4 u5) u2 u3 u5 + E^(u1 u2 u3 u4 u5) u1 u2^2 u3^2 u4 u5^2
    const Real H14 = (exp_val * u2 * u3 * u5)
            + (exp_val * u1 * pow(u2, static_cast<Real>(2.)) * pow(u3, static_cast<Real>(2.)) * u4
                    * pow(u5, static_cast<Real>(2.)));
    // E^(u1 u2 u3 u4 u5) u2 u3 u4 + E^(u1 u2 u3 u4 u5) u1 u2^2 u3^2 u4^2 u5
    const Real H15 = (exp_val * u2 * u3 * u4)
            + (exp_val * u1 * pow(u2, static_cast<Real>(2.)) * pow(u3, static_cast<Real>(2.))
                    * pow(u4, static_cast<Real>(2.)) * u5);
    // column 2
    // -9. u2^4 - 6. u2 (1 + u1^3 + u2^3) + E^(u1 u2 u3 u4 u5) u1^2 u3^2 u4^2 u5^2
    const Real H22 = (exp_val * pow(u1, static_cast<Real>(2.)) * pow(u3, static_cast<Real>(2.))
            * pow(u4, static_cast<Real>(2.)) * pow(u5, static_cast<Real>(2.)))
            - (static_cast<Real>(9.) * pow(u2, static_cast<Real>(4.)))
            - (static_cast<Real>(6.) * u2
                    * (static_cast<Real>(1.) + pow(u1, static_cast<Real>(3.)) + pow(u2, static_cast<Real>(3.))));
    // E^(u1 u2 u3 u4 u5) u1 u4 u5 + E^(u1 u2 u3 u4 u5) u1^2 u2 u3 u4^2 u5^2
    const Real H23 = (exp_val * u1 * u4 * u5)
            + (exp_val * pow(u1, static_cast<Real>(2.)) * u2 * u3 * pow(u4, static_cast<Real>(2.))
                    * pow(u5, static_cast<Real>(2.)));
    // E^(u1 u2 u3 u4 u5) u1 u3 u5 + E^(u1 u2 u3 u4 u5) u1^2 u2 u3^2 u4 u5^2
    const Real H24 = (exp_val * u1 * u3 * u5)
            + (exp_val * pow(u1, static_cast<Real>(2.)) * u2 * pow(u3, static_cast<Real>(2.)) * u4
                    * pow(u5, static_cast<Real>(2.)));
    // E^(u1 u2 u3 u4 u5) u1 u3 u4 + E^(u1 u2 u3 u4 u5) u1^2 u2 u3^2 u4^2 u5
    const Real H25 = (exp_val * u1 * u3 * u4)
            + (exp_val * pow(u1, static_cast<Real>(2.)) * u2 * pow(u3, static_cast<Real>(2.))
                    * pow(u4, static_cast<Real>(2.)) * u5);
    // column 3
    // E^(u1 u2 u3 u4 u5) u1^2 u2^2 u4^2 u5^2
    const Real H33 = exp_val * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.))
            * pow(u4, static_cast<Real>(2.)) * pow(u5, static_cast<Real>(2.));
    // E^(u1 u2 u3 u4 u5) u1 u2 u5 + E^(u1 u2 u3 u4 u5) u1^2 u2^2 u3 u4 u5^2
    const Real H34 = (exp_val * u1 * u2 * u5)
            + (exp_val * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.)) * u3 * u4
                    * pow(u5, static_cast<Real>(2.)));
    // E^(u1 u2 u3 u4 u5) u1 u2 u4 + E^(u1 u2 u3 u4 u5) u1^2 u2^2 u3 u4^2 u5
    const Real H35 = (exp_val * u1 * u2 * u4)
            + (exp_val * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.)) * u3
                    * pow(u4, static_cast<Real>(2.)) * u5);
    // column 4
    // E^(u1 u2 u3 u4 u5) u1^2 u2^2 u3^2 u5^2
    const Real H44 = exp_val * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.))
            * pow(u3, static_cast<Real>(2.)) * pow(u5, static_cast<Real>(2.));
    // E^(u1 u2 u3 u4 u5) u1 u2 u3 + E^(u1 u2 u3 u4 u5) u1^2 u2^2 u3^2 u4 u5
    const Real H45 = (exp_val * u1 * u2 * u3)
            + (exp_val * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.))
                    * pow(u3, static_cast<Real>(2.)) * u4 * u5);
    // column 5
    // E^(u1 u2 u3 u4 u5) u1^2 u2^2 u3^2 u4^2
    const Real H55 = exp_val * pow(u1, static_cast<Real>(2.)) * pow(u2, static_cast<Real>(2.))
            * pow(u3, static_cast<Real>(2.)) * pow(u4, static_cast<Real>(2.));
    // Hessian-vector product
    output_[0] = H11 * vector_[0] + H12 * vector_[1] + H13 * vector_[2] + H14 * vector_[3] + H15 * vector_[4];
    output_[1] = H12 * vector_[0] + H22 * vector_[1] + H23 * vector_[2] + H24 * vector_[3] + H25 * vector_[4];
    output_[2] = H13 * vector_[0] + H23 * vector_[1] + H33 * vector_[2] + H34 * vector_[3] + H35 * vector_[4];
    output_[3] = H14 * vector_[0] + H24 * vector_[1] + H34 * vector_[2] + H44 * vector_[3] + H45 * vector_[4];
    output_[4] = H15 * vector_[0] + H25 * vector_[1] + H35 * vector_[2] + H45 * vector_[3] + H55 * vector_[4];
}

}
