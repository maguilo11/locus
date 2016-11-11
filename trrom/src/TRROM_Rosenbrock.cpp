/*
 * TRROM_Rosenbrock.cpp
 *
 *  Created on: Aug 14, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Vector.hpp"
#include "TRROM_Rosenbrock.hpp"

namespace trrom
{

Rosenbrock::Rosenbrock()
{
}

Rosenbrock::~Rosenbrock()
{
}

double Rosenbrock::value(const double & tolerance_,
                         const trrom::Vector<double> & control_,
                         bool & inexactness_violated_)
{
    double value;
    value = static_cast<double>(100.) * std::pow((control_[1] - control_[0] * control_[0]), static_cast<double>(2.))
            + std::pow(static_cast<double>(1.) - control_[0], static_cast<double>(2.));
    return (value);
}

void Rosenbrock::gradient(const double & tolerance_,
                          const trrom::Vector<double> & control_,
                          trrom::Vector<double> & gradient_,
                          bool & inexactness_violated_)
{
    gradient_[0] = static_cast<double>(-400.) * (control_[1] - std::pow(control_[0], 2.)) * control_[0]
            + static_cast<double>(2.) * control_[0] - static_cast<double>(2.);
    gradient_[1] = static_cast<double>(200.) * (control_[1] - std::pow(control_[0], static_cast<double>(2.)));
}

void Rosenbrock::hessian(const double & tolerance_,
                         const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & hess_times_vec_,
                         bool & inexactness_violated_)
{
    hess_times_vec_[0] = ((static_cast<double>(2.)
            - static_cast<double>(400.) * (control_[1] - std::pow(control_[0], static_cast<double>(2.)))
            + static_cast<double>(800.) * std::pow(control_[0], static_cast<double>(2.))) * vector_[0])
            - (static_cast<double>(400.) * control_[0] * vector_[1]);
    hess_times_vec_[1] = (static_cast<double>(-400.) * control_[0] * vector_[0])
            + (static_cast<double>(200.) * vector_[1]);
}

}
