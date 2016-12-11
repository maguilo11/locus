/*
 * DOTk_ZakharovObjective.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "DOTk_ZakharovObjective.hpp"

namespace dotk
{

DOTk_ZakharovObjective::DOTk_ZakharovObjective(const dotk::Vector<Real> & input_) :
        m_Data(input_.clone())
{
}

DOTk_ZakharovObjective::~DOTk_ZakharovObjective()
{
}

Real DOTk_ZakharovObjective::value(const dotk::Vector<Real> & primal_)
{
    Real primal_dot_primal = primal_.dot(primal_);
    Real k_dot_primal = primal_.dot(*m_Data);

    Real value = primal_dot_primal + std::pow(k_dot_primal, 2.) / static_cast<Real>(4.)
            + std::pow(k_dot_primal, 4.) / static_cast<Real>(16.);

    return (value);
}

void DOTk_ZakharovObjective::gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_)
{
    Real k_dot_primal = primal_.dot(*m_Data);
    Real coeff = static_cast<Real>(0.25) * (static_cast<Real>(2.) * k_dot_primal + std::pow(k_dot_primal, 3.));

    output_.axpy(2., primal_);
    output_.axpy(coeff, *m_Data);
}

void DOTk_ZakharovObjective::hessian(const dotk::Vector<Real> & primal_,
                                     const dotk::Vector<Real> & vector_,
                                     dotk::Vector<Real> & output_)
{
    Real k_dot_primal = primal_.dot(*m_Data);
    Real k_dot_delta_primal = vector_.dot(*m_Data);
    Real coeff = static_cast<Real>(0.25) * (static_cast<Real>(2.) + static_cast<Real>(3.) * std::pow(k_dot_primal, 2.))
            * k_dot_delta_primal;

    output_.axpy(coeff, *m_Data);
    output_.axpy(2., vector_);
}

}
