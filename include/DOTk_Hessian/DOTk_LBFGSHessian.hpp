/*
 * DOTk_LBFGSHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LBFGSHESSIAN_HPP_
#define DOTK_LBFGSHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_LBFGSHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LBFGSHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                      size_t max_secant_storage_);
    virtual ~DOTk_LBFGSHessian();

    const std::tr1::shared_ptr<std::vector<Real> > & getDeltaGradPrimalInnerProductStorage() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaGradStorage(size_t at_) const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaPrimalStorage(size_t at_) const;

    void getHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & hess_times_vec_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng>& mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    std::tr1::shared_ptr<std::vector<Real> > m_RhoStorage;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;

    std::tr1::shared_ptr<dotk::matrix<Real> > m_MatrixA;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_MatrixB;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;


private:
    DOTk_LBFGSHessian(const dotk::DOTk_LBFGSHessian &);
    dotk::DOTk_LBFGSHessian operator=(const dotk::DOTk_LBFGSHessian &);
};

}

#endif /* DOTK_LBFGSHESSIAN_HPP_ */
