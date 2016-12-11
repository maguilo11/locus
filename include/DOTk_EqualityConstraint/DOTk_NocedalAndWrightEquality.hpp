/*
 * DOTk_NocedalAndWrightEquality.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NOCEDALANDWRIGHTEQUALITY_HPP_
#define DOTK_NOCEDALANDWRIGHTEQUALITY_HPP_

#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_NocedalAndWrightEquality : public dotk::DOTk_EqualityConstraint<Real>
{
public:
    DOTk_NocedalAndWrightEquality();
    virtual ~DOTk_NocedalAndWrightEquality();

    virtual void residual(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & residual_);
    virtual void jacobian(const dotk::Vector<Real> & primal_,
                          const dotk::Vector<Real> & vector_,
                          dotk::Vector<Real> & output_);
    virtual void adjointJacobian(const dotk::Vector<Real> & primal_,
                                 const dotk::Vector<Real> & dual_,
                                 dotk::Vector<Real> & output_);
    virtual void hessian(const dotk::Vector<Real> & primal_,
                         const dotk::Vector<Real> & dual_,
                         const dotk::Vector<Real> & vector_,
                         dotk::Vector<Real> & output_);

private:
    DOTk_NocedalAndWrightEquality(const dotk::DOTk_NocedalAndWrightEquality&);
    dotk::DOTk_NocedalAndWrightEquality operator=(const dotk::DOTk_NocedalAndWrightEquality&);
};

}

#endif /* DOTK_NOCEDALANDWRIGHTEQUALITY_HPP_ */
