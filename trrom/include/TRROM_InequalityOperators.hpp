/*
 * TRROM_InequalityOperators.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_INEQUALITYOPERATORS_HPP_
#define TRROM_INEQUALITYOPERATORS_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class InequalityOperators
{
public:
    virtual ~InequalityOperators()
    {
    }

    /*! h(\mathbf{u}(\mathbf{z}),\mathbf{z}) \equiv value(\mathbf{u}(\mathbf{z}),\mathbf{z}) - bound \leq 0 */
    virtual double bound() = 0;
    virtual double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_) = 0;

    /*! \partial{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{z}} */
    virtual void partialDerivativeState(const trrom::Vector<double> & state_,
                                        const trrom::Vector<double> & control_,
                                        const trrom::Vector<double> & output_) = 0;
    /*! \partial{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{u}} */
    virtual void partialDerivativeControl(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          trrom::Vector<double> & output_) = 0;

    /*! \partial^2{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{z}\partial\mathbf{u}} */
    virtual void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
    /*! \partial^2{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{z}\partial\mathbf{z}} */
    virtual void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & vector_,
                                                 trrom::Vector<double> & output_) = 0;
    /*! \partial^2{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{u}\partial\mathbf{u}} */
    virtual void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                             const trrom::Vector<double> & control_,
                                             const trrom::Vector<double> & vector_,
                                             trrom::Vector<double> & output_) = 0;
    /*! \partial^2{h(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{u}\partial\mathbf{z}} */
    virtual void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
};

}

#endif /* TRROM_INEQUALITYOPERATORS_HPP_ */
