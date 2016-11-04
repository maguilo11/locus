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

template<class Type>
class vector;

class DOTk_NocedalAndWrightEquality : public dotk::DOTk_EqualityConstraint<Real>
{
public:
    DOTk_NocedalAndWrightEquality();
    virtual ~DOTk_NocedalAndWrightEquality();

    virtual void residual(const dotk::vector<Real> & primal_, dotk::vector<Real> & residual_);
    virtual void jacobian(const dotk::vector<Real> & primal_,
                          const dotk::vector<Real> & vector_,
                          dotk::vector<Real> & output_);
    virtual void adjointJacobian(const dotk::vector<Real> & primal_,
                                 const dotk::vector<Real> & dual_,
                                 dotk::vector<Real> & output_);
    virtual void hessian(const dotk::vector<Real> & primal_,
                         const dotk::vector<Real> & dual_,
                         const dotk::vector<Real> & vector_,
                         dotk::vector<Real> & output_);

private:
    DOTk_NocedalAndWrightEquality(const dotk::DOTk_NocedalAndWrightEquality&);
    dotk::DOTk_NocedalAndWrightEquality operator=(const dotk::DOTk_NocedalAndWrightEquality&);
};

}

#endif /* DOTK_NOCEDALANDWRIGHTEQUALITY_HPP_ */
