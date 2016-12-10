/*
 * TRROM_Radius.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "TRROM_Radius.hpp"
#include "TRROM_Vector.hpp"

namespace trrom
{

Radius::Radius()
{
}

Radius::~Radius()
{
}

/// Inequality constraint: h(\mathbf{z}) \equiv value(\mathbf{z}) - bound \leq 0
double Radius::bound()
{
    return (1);
}

double Radius::value(const trrom::Vector<double> & control_)
{
    double value = std::pow(control_[0], 2.) + std::pow(control_[1], 2.);
    return (value);
}

/// Gradient operator: \partial{h(\mathbf{z})}{\partial\mathbf{z}}
void Radius::gradient(const trrom::Vector<double> & control_, trrom::Vector<double> & output_)
{
    output_[0] = static_cast<double>(2.) * control_[0];
    output_[1] = static_cast<double>(2.) * control_[1];
}

/// Application of vector to hessian operator: \partial^2{h(\mathbf{z})}{\partial\mathbf{z}^2}
void Radius::hessian(const trrom::Vector<double> & control_,
                     const trrom::Vector<double> & vector_,
                     trrom::Vector<double> & output_)
{
    output_[0] = static_cast<double>(2.) * vector_[0];
    output_[1] = static_cast<double>(2.) * vector_[1];
}

}
