/*
 * DOTk_LDFPHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LDFPHESSIAN_HPP_
#define DOTK_LDFPHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_LDFPHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LDFPHessian(const dotk::Vector<Real> & vector_, size_t max_secant_storage_);
    virtual ~DOTk_LDFPHessian();

    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGradStorage(size_t at_) const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimalStorage(size_t at_) const;

    void getHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                    const std::shared_ptr<dotk::Vector<Real> > & hess_times_vec_);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    std::vector<Real> m_Alpha;
    std::vector<Real> m_RhoStorage;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;


private:
    DOTk_LDFPHessian(const dotk::DOTk_LDFPHessian &);
    dotk::DOTk_LDFPHessian operator=(const dotk::DOTk_LDFPHessian &);
};

}

#endif /* DOTK_LDFPHESSIAN_HPP_ */
