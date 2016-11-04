/*
 * DOTk_GcmmaTestInequalityConstraint.hpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GCMMATESTINEQUALITYCONSTRAINT_HPP_
#define DOTK_GCMMATESTINEQUALITYCONSTRAINT_HPP_

#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

template<class Type>
class vector;

class DOTk_GcmmaTestInequalityConstraint : public dotk::DOTk_InequalityConstraint<Real>
{
public:
    DOTk_GcmmaTestInequalityConstraint();
    virtual ~DOTk_GcmmaTestInequalityConstraint();

    Real residual(const dotk::vector<Real> & primal_);

    Real bound();
    Real value(const dotk::vector<Real> & primal_);
    void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & gradient_);

private:
    Real m_Constant;

private:
    DOTk_GcmmaTestInequalityConstraint(const dotk::DOTk_GcmmaTestInequalityConstraint &);
    dotk::DOTk_GcmmaTestInequalityConstraint & operator=(const dotk::DOTk_GcmmaTestInequalityConstraint &);
};

}

#endif /* DOTK_GCMMATESTINEQUALITYCONSTRAINT_HPP_ */
