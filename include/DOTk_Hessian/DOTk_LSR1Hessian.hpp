/*
 * DOTk_LSR1Hessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LSR1HESSIAN_HPP_
#define DOTK_LSR1HESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_LSR1Hessian: public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LSR1Hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_, size_t max_secant_storage_);
    virtual ~DOTk_LSR1Hessian();

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaGradStorage(size_t at_) const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaPrimalStorage(size_t at_) const;

    void unrollingSR1(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                      const std::tr1::shared_ptr<dotk::Vector<Real> > & hess_times_vector_);
    void getHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & hess_times_vector_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;

    std::tr1::shared_ptr<dotk::matrix<Real> > m_MatrixA;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;

private:
    DOTk_LSR1Hessian(const dotk::DOTk_LSR1Hessian &);
    DOTk_LSR1Hessian operator=(const dotk::DOTk_LSR1Hessian &);
};

}

#endif /* DOTK_LSR1HESSIAN_HPP_ */
