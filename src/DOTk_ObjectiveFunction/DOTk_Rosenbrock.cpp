/*
 * DOTk_Rosenbrock.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "DOTk_Rosenbrock.hpp"

namespace dotk
{

DOTk_Rosenbrock::DOTk_Rosenbrock() :
        dotk::DOTk_ObjectiveFunction<Real>()
{
}

DOTk_Rosenbrock::~DOTk_Rosenbrock()
{
}

Real DOTk_Rosenbrock::value(const dotk::Vector<Real> & primal_)
{
    Real value;
    value = static_cast<Real>(100.) * std::pow((primal_[1] - primal_[0] * primal_[0]), static_cast<Real>(2.))
            + std::pow(static_cast<Real>(1.) - primal_[0], static_cast<Real>(2.));

    return (value);
}

void DOTk_Rosenbrock::value(const std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > & primal_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & fval_)
{
    size_t number_controls = fval_->size();
    for(size_t index = 0; index < number_controls; ++ index)
    {
        (*fval_)[index] = this->value(*primal_[index]);
    }
}

void DOTk_Rosenbrock::value(const std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > & primal_plus_,
                            const std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > & primal_minus_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & values_plus_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & values_minus_)
{
    size_t number_controls = values_plus_->size();
    for(size_t index = 0; index < number_controls; ++ index)
    {
        // Compute forward perturbation of objective function
        (*values_plus_)[index] = this->value(*primal_plus_[index]);

        // Compute backward perturbation of objective function
        (*values_minus_)[index] = this->value(*primal_minus_[index]);
    }
}

void DOTk_Rosenbrock::gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & gradient_)
{
    gradient_[0] = static_cast<Real>(-400.) * (primal_[1] - std::pow(primal_[0], 2.)) * primal_[0]
            + static_cast<Real>(2.) * primal_[0] - static_cast<Real>(2.);
    gradient_[1] = static_cast<Real>(200.) * (primal_[1] - std::pow(primal_[0], static_cast<Real>(2.)));
}

void DOTk_Rosenbrock::hessian(const dotk::Vector<Real> & primal_,
                              const dotk::Vector<Real> & vector_,
                              dotk::Vector<Real> & output_)
{
    output_[0] = ((static_cast<Real>(2.)
            - static_cast<Real>(400.) * (primal_[1] - std::pow(primal_[0], static_cast<Real>(2.)))
            + static_cast<Real>(800.) * std::pow(primal_[0], static_cast<Real>(2.))) * vector_[0])
            - (static_cast<Real>(400.) * primal_[0] * vector_[1]);
    output_[1] = (static_cast<Real>(-400.) * primal_[0] * vector_[0]) + (static_cast<Real>(200.) * vector_[1]);
}

}
