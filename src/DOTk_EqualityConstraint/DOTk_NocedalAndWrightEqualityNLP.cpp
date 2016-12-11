/*
 * DOTk_NocedalAndWrightEqualityNLP.cpp
 *
 *  Created on: Mar 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include "DOTk_NocedalAndWrightEqualityNLP.hpp"

namespace dotk
{

DOTk_NocedalAndWrightEqualityNLP::DOTk_NocedalAndWrightEqualityNLP()
{
}

DOTk_NocedalAndWrightEqualityNLP::~DOTk_NocedalAndWrightEqualityNLP()
{
}

void DOTk_NocedalAndWrightEqualityNLP::residual(const dotk::Vector<Real> & state_,
                                                const dotk::Vector<Real> & control_,
                                                dotk::Vector<Real> & residual_)
{
    // C1 = u1^2 + u2^2 + u3^2 + u4^2 + u5^2 - 10
    residual_[0] = pow(state_[0], static_cast<Real>(2.)) + pow(state_[1], static_cast<Real>(2.))
            + pow(state_[2], static_cast<Real>(2.)) + pow(state_[3], 2.) + pow(state_[4], 2.) - static_cast<Real>(10.);

    // C2 = u2 * u3 - 5. * u4 * u5
    residual_[1] = state_[1] * state_[2] - static_cast<Real>(5.) * state_[3] * state_[4];

    // C3 = u1^3 + u2^3 + 1.
    residual_[2] = pow(state_[0], static_cast<Real>(3.)) + pow(state_[1], static_cast<Real>(3.))
            + static_cast<Real>(1.);
}

void DOTk_NocedalAndWrightEqualityNLP::partialDerivativeState(const dotk::Vector<Real> & state_,
                                                              const dotk::Vector<Real> & control_,
                                                              const dotk::Vector<Real> & delta_state_,
                                                              dotk::Vector<Real> & jacobian_times_delta_state_)
{
    // |  2*u1    2*u2   2*u3   2*u4   2*u5 |
    // |   0       u3     u2   -5*u5  -5*u4 |
    // | 3*u1^2  3*u2^2   0      0      0   |

    jacobian_times_delta_state_[0] = static_cast<Real>(2.) * state_[0] * delta_state_[0]
            + static_cast<Real>(2.) * state_[1] * delta_state_[1] + static_cast<Real>(2.) * state_[2] * delta_state_[2]
            + static_cast<Real>(2.) * state_[3] * delta_state_[3] + static_cast<Real>(2.) * state_[4] * delta_state_[4];

    jacobian_times_delta_state_[1] = state_[2] * delta_state_[1] + state_[1] * delta_state_[2]
            - static_cast<Real>(5.) * state_[4] * delta_state_[3] - static_cast<Real>(5.) * state_[3] * delta_state_[4];

    jacobian_times_delta_state_[2] = static_cast<Real>(3.) * pow(state_[0], 2.) * delta_state_[0]
            + static_cast<Real>(3.) * pow(state_[1], static_cast<Real>(2.)) * delta_state_[1];
}

void DOTk_NocedalAndWrightEqualityNLP::partialDerivativeControl(const dotk::Vector<Real> & state_,
                                                                const dotk::Vector<Real> & control_,
                                                                const dotk::Vector<Real> & vector_,
                                                                dotk::Vector<Real> & output_)
{
    return;
}

void DOTk_NocedalAndWrightEqualityNLP::adjointPartialDerivativeState(const dotk::Vector<Real> & state_,
                                                                     const dotk::Vector<Real> & control_,
                                                                     const dotk::Vector<Real> & dual_,
                                                                     dotk::Vector<Real> & output_)
{
    // | 2*u1    0    3*u1^2 |
    // | 2*u2    u3   3*u2^2 |
    // | 2*u3    u2     0    |
    // | 2*u4  -5*u5    0    |
    // | 2*u5  -5*u4    0    |

    output_[0] = static_cast<Real>(2.) * state_[0] * dual_[0]
            + static_cast<Real>(3.) * pow(state_[0], static_cast<Real>(2.)) * dual_[2];

    output_[1] = static_cast<Real>(2.) * state_[1] * dual_[0] + state_[2] * dual_[1]
            + static_cast<Real>(3.) * pow(state_[1], static_cast<Real>(2.)) * dual_[2];

    output_[2] = static_cast<Real>(2.) * state_[2] * dual_[0] + state_[1] * dual_[1];

    output_[3] = static_cast<Real>(2.) * state_[3] * dual_[0] - static_cast<Real>(5.) * state_[4] * dual_[1];

    output_[4] = static_cast<Real>(2.) * state_[4] * dual_[0] - static_cast<Real>(5.) * state_[3] * dual_[1];
}

void DOTk_NocedalAndWrightEqualityNLP::adjointPartialDerivativeControl(const dotk::Vector<Real> & state_,
                                                                       const dotk::Vector<Real> & control_,
                                                                       const dotk::Vector<Real> & dual_,
                                                                       dotk::Vector<Real> & output_)
{
    return;
}

void DOTk_NocedalAndWrightEqualityNLP::partialDerivativeStateState(const dotk::Vector<Real> & state_,
                                                                   const dotk::Vector<Real> & control_,
                                                                   const dotk::Vector<Real> & dual_,
                                                                   const dotk::Vector<Real> & delta_state_,
                                                                   dotk::Vector<Real> & vector_)
{
    const Real du1 = delta_state_[0];
    const Real du2 = delta_state_[1];
    const Real du3 = delta_state_[2];
    const Real du4 = delta_state_[3];
    const Real du5 = delta_state_[4];
    const Real lambda1 = dual_[0];
    const Real lambda2 = dual_[1];
    const Real lambda3 = dual_[2];

    vector_[0] = static_cast<Real>(2.) * lambda1 * du1 + static_cast<Real>(6.) * state_[0] * lambda3 * du1;

    vector_[1] = static_cast<Real>(2.) * lambda1 * du2 + static_cast<Real>(6.) * state_[1] * lambda3 * du2
            + lambda2 * du3;

    vector_[2] = lambda2 * du2 + static_cast<Real>(2.) * lambda1 * du3;

    vector_[3] = static_cast<Real>(2.) * lambda1 * du4 - static_cast<Real>(5.) * lambda2 * du5;

    vector_[4] = static_cast<Real>(-5.) * lambda2 * du4 + static_cast<Real>(2.) * lambda1 * du5;
}

void DOTk_NocedalAndWrightEqualityNLP::partialDerivativeStateControl(const dotk::Vector<Real> & state_,
                                                                     const dotk::Vector<Real> & control_,
                                                                     const dotk::Vector<Real> & dual_,
                                                                     const dotk::Vector<Real> & vector_,
                                                                     dotk::Vector<Real> & output_)
{
    return;
}

void DOTk_NocedalAndWrightEqualityNLP::partialDerivativeControlControl(const dotk::Vector<Real> & state_,
                                                                       const dotk::Vector<Real> & control_,
                                                                       const dotk::Vector<Real> & dual_,
                                                                       const dotk::Vector<Real> & vector_,
                                                                       dotk::Vector<Real> & output_)
{
    return;
}

void DOTk_NocedalAndWrightEqualityNLP::partialDerivativeControlState(const dotk::Vector<Real> & state_,
                                                                     const dotk::Vector<Real> & control_,
                                                                     const dotk::Vector<Real> & dual_,
                                                                     const dotk::Vector<Real> & delta_state_,
                                                                     dotk::Vector<Real> & vector_)
{
    return;
}

}
