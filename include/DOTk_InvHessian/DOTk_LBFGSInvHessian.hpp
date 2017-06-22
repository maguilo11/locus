/*
 * DOTk_LBFGSInvHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LBFGSINVHESSIAN_HPP_
#define DOTK_LBFGSINVHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_LBFGSInvHessian: public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LBFGSInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_, size_t max_secant_storage_);
    virtual ~DOTk_LBFGSInvHessian();

    const std::shared_ptr<std::vector<Real> > & getDeltaGradPrimalInnerProductStorage() const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGradStorage(size_t at_) const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimalStorage(size_t at_) const;

    void getInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & inv_hess_times_vector_);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_);

private:
    std::vector<Real> m_Alpha;
    std::shared_ptr<std::vector<Real> > m_RhoStorage;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;

    std::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;

private:
    DOTk_LBFGSInvHessian(const dotk::DOTk_LBFGSInvHessian &);
    DOTk_LBFGSInvHessian operator=(const dotk::DOTk_LBFGSInvHessian &);
};

}

#endif /* DOTK_LBFGSINVHESSIAN_HPP_ */
