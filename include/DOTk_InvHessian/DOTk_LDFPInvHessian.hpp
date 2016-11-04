/*
 * DOTk_LDFPInvHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_LDFPINVHESSIAN_HPP_
#define DOTK_LDFPINVHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;
template<class Type>
class matrix;

class DOTk_LDFPInvHessian: public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LDFPInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t max_secant_storage_);
    virtual ~DOTk_LDFPInvHessian();

    const std::tr1::shared_ptr<std::vector<Real> > & getDeltaGradPrimalInnerProductStorage() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaGradStorage(size_t at_) const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaPrimalStorage(size_t at_) const;

    void getInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & inv_hess_times_vector_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_);

private:
    std::tr1::shared_ptr<std::vector<Real> > m_RhoStorage;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaGradient;

    std::tr1::shared_ptr<dotk::matrix<Real> > m_MatrixA;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_MatrixB;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;

private:
    DOTk_LDFPInvHessian(const dotk::DOTk_LDFPInvHessian &);
    DOTk_LDFPInvHessian operator=(const dotk::DOTk_LDFPInvHessian &);
};

}

#endif /* DOTK_LDFPINVHESSIAN_HPP_ */
