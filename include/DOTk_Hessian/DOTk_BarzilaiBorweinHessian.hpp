/*
 * DOTk_BarzilaiBorweinHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BARZILAIBORWEINHESSIAN_HPP_
#define DOTK_BARZILAIBORWEINHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_BarzilaiBorweinHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_BarzilaiBorweinHessian(const dotk::Vector<Real> & vector_);
    virtual ~DOTk_BarzilaiBorweinHessian();

    void computeDeltaPrimal(const std::shared_ptr<dotk::Vector<Real> > & new_primal_,
                            const std::shared_ptr<dotk::Vector<Real> > & old_primal_);
    void computeDeltaGradient(const std::shared_ptr<dotk::Vector<Real> > & new_gradient_,
                              const std::shared_ptr<dotk::Vector<Real> > & old_gradient_);
    void getHessian(const std::shared_ptr<dotk::Vector<Real> > & vec_,
                    const std::shared_ptr<dotk::Vector<Real> > & mat_times_vec_);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vec_,
                       const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;


private:
    DOTk_BarzilaiBorweinHessian(const dotk::DOTk_BarzilaiBorweinHessian &);
    dotk::DOTk_BarzilaiBorweinHessian operator=(const dotk::DOTk_BarzilaiBorweinHessian &);
};


}

#endif /* DOTK_BARZILAIBORWEINHESSIAN_HPP_ */
