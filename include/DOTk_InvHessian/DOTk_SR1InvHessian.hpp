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

template<class Type>
class vector;

class DOTk_SR1InvHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_SR1InvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_SR1InvHessian();

    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaGrad() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaPrimal() const;

    void getInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & inv_hess_times_vector_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InvHessTimesVec;

private:
    DOTk_SR1InvHessian(const dotk::DOTk_SR1InvHessian &);
    DOTk_SR1InvHessian operator=(const dotk::DOTk_SR1InvHessian &);
};

}

#endif /* DOTK_SR1INVHESSIAN_HPP_ */
