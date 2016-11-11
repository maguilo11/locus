/*
 * TRROM_InequalityTypeNP.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_INEQUALITYTYPENP_HPP_
#define TRROM_INEQUALITYTYPENP_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class InequalityTypeNP
{
public:
    virtual ~InequalityTypeNP()
    {
    }

    /// Inequality constraint: h(\mathbf{u}(\mathbf{z}),\mathbf{z}) \equiv
    /// value(\mathbf{u}(\mathbf{z}),\mathbf{z}) - bound \leq 0
    virtual double bound() = 0;
    virtual double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_) = 0;

    /// Gradient operator: \partial{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{z}}
    virtual void gradient(const trrom::Vector<double> & state_,
                          const trrom::Vector<double> & control_,
                          trrom::Vector<double> & output_) = 0;

    /// Application of vector to hessian operator: \partial^2{h(\mathbf{z})}{\partial\mathbf{z}^2}
    virtual void hessian(const trrom::Vector<double> & state_,
                         const trrom::Vector<double> & control_,
                         const trrom::Vector<double> & vector_,
                         trrom::Vector<double> & output_) = 0;
};

}

#endif /* TRROM_INEQUALITYTYPENP_HPP_ */
