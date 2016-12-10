/*
 * TRROM_Radius.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_INCLUDE_RADIUS_HPP_
#define TRROM_INCLUDE_RADIUS_HPP_

#include "TRROM_InequalityTypeLP.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Radius : public trrom::InequalityTypeLP
{
public:
    Radius();
    virtual ~Radius();

    /// Inequality constraint: h(\mathbf{z}) \equiv value(\mathbf{z}) - bound \leq 0
    virtual double bound();
    virtual double value(const trrom::Vector<double> & control_);
    /// Gradient operator: \partial{h(\mathbf{z})}{\partial\mathbf{z}}
    virtual void gradient(const trrom::Vector<double> & control_, trrom::Vector<double> & output_);
    /// Application of vector to hessian operator: \partial^2{h(\mathbf{z})}{\partial\mathbf{z}^2}
    virtual void hessian(const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & output_);

private:
    Radius(const trrom::Radius &);
    trrom::Radius & operator=(const trrom::Radius &);
};

}

#endif /* TRROM_INCLUDE_RADIUS_HPP_ */
