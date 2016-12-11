/*
 * DOTk_SR1InvHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_SR1INVHESSIAN_HPP_
#define DOTK_SR1INVHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_SR1InvHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_SR1InvHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_);
    virtual ~DOTk_SR1InvHessian();

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaGrad() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaPrimal() const;

    void getInvHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & inv_hess_times_vector_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_InvHessTimesVec;

private:
    DOTk_SR1InvHessian(const dotk::DOTk_SR1InvHessian &);
    DOTk_SR1InvHessian operator=(const dotk::DOTk_SR1InvHessian &);
};

}

#endif /* DOTK_SR1INVHESSIAN_HPP_ */
