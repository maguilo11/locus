/*
 * TRROM_InequalityTypeLP.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_INEQUALITYTYPELP_HPP_
#define TRROM_INEQUALITYTYPELP_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class InequalityTypeLP
{
public:
    virtual ~InequalityTypeLP()
    {
    }

    /// Inequality constraint: h(\mathbf{z}) \equiv value(\mathbf{z}) - bound \leq 0
    virtual double bound() = 0;
    virtual double value(const trrom::Vector<double> & control_) = 0;

    /// Gradient operator: \partial{h(\mathbf{z})}{\partial\mathbf{z}}
    virtual void gradient(const trrom::Vector<double> & control_, trrom::Vector<double> & output_) = 0;

    /// Application of vector to hessian operator: \partial^2{h(\mathbf{z})}{\partial\mathbf{z}^2}
    virtual void hessian(const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & output_) = 0;
};

}

#endif /* TRROM_INEQUALITYTYPELP_HPP_ */
