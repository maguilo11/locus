/*
 * DOTk_BFGSInvHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_BFGSINVHESSIAN_HPP_
#define DOTK_BFGSINVHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_BFGSInvHessian: public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_BFGSInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_);
    virtual ~DOTk_BFGSInvHessian();

    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGrad() const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimal() const;

    void getInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & inv_hess_times_vector_);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_);

private:
    std::shared_ptr<dotk::Vector<Real> > mDeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > mDeltaGradient;
    std::shared_ptr<dotk::Vector<Real> > m_InvHessTimesVec;

private:
    DOTk_BFGSInvHessian(const dotk::DOTk_BFGSInvHessian &);
    DOTk_BFGSInvHessian operator=(const dotk::DOTk_BFGSInvHessian &);
};

}

#endif /* DOTK_BFGSINVHESSIAN_HPP_ */
