/*
 * TRROM_Circle.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "TRROM_Vector.hpp"
#include "TRROM_Circle.hpp"

namespace trrom
{

Circle::Circle()
{
}

Circle::~Circle()
{
}

double Circle::value(const double & tolerance_, const trrom::Vector<double> & control_, bool & inexactness_violated_)
{
    /// \left(\mathbf{z}(0) - 1.\right)^2 + 2\left(\mathbf{z}(1) - 2\right)^2
    double alpha = control_[0] - static_cast<double>(1.);
    double beta = control_[1] - 2;
    beta = static_cast<double>(2.) * std::pow(beta, 2.);
    double output = std::pow(alpha, 2.) + beta;

    return (output);
}

void Circle::gradient(const double & tolerance_,
                      const trrom::Vector<double> & control_,
                      trrom::Vector<double> & gradient_,
                      bool & inexactness_violated_)
{
    gradient_[0] = static_cast<double>(2.) * (control_[0] - static_cast<double>(1.));
    gradient_[1] = static_cast<double>(4.) * (control_[1] - static_cast<double>(2.));
}

void Circle::hessian(const double & tolerance_,
                     const trrom::Vector<double> & control_,
                     const trrom::Vector<double> & vector_,
                     trrom::Vector<double> & hess_times_vec_,
                     bool & inexactness_violated_)
{
    hess_times_vec_[0] = static_cast<double>(2.) * vector_[0];
    hess_times_vec_[1] = static_cast<double>(4.) * vector_[1];
}

}
